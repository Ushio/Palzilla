#include "minimum_lbvh.h"
#include "helper_math.h"
#include "camera.h"
#include "sobol.h"
#include "pk.h"

using namespace minimum_lbvh;

constexpr float PI = 3.14159265358979323846f;

__device__ uint32_t packRGBA( float4 color )
{
    int4 i4 = make_int4(color * 255.0f + make_float4(0.5f));
    i4 = clamp(i4, 0, 255);
    return (i4.z << 16) | (i4.y << 8) | i4.x;
}

__device__ float3 sampleHemisphereCosWeighted(float xi_0, float xi_1)
{
    float phi = xi_0 * 2.0f * PI;
    float r = sqrtf(xi_1);

    // uniform in a circle
    float x = cosf(phi) * r;
    float z = sinf(phi) * r;

    // project to hemisphere
    float y = sqrtf(fmax(1.0f - r * r, 0.0f));
    return { x, y, z };
}

// Building an Orthonormal Basis, Revisited
__device__ void GetOrthonormalBasis(float3 zaxis, float3* xaxis, float3* yaxis) {
    const float sign = copysignf(1.0f, zaxis.z);
    const float a = -1.0f / (sign + zaxis.z);
    const float b = zaxis.x * zaxis.y * a;
    *xaxis = { 1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x };
    *yaxis = { b, sign + zaxis.y * zaxis.y * a, -zaxis.y };
}

extern "C" __global__ void normal(uint32_t *pixels, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles )
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (xi < imageSize.x && yi < imageSize.y)
    {
        int pixel = xi + yi * imageSize.x;

        float3 ro, rd;
        rayGenerator.shoot(&ro, &rd, (float)xi / imageSize.x, (float)yi / imageSize.y);

        Hit hit;
        intersect_stackfree(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd));
        if (hit.t != MINIMUM_LBVH_FLT_MAX)
        {
            float3 n = normalize(hit.ng);
            float3 color = (n + make_float3(1.0f)) * 0.5f;
            pixels[pixel] = packRGBA({ color.x, color.y, color.z, 1.0f });
        }
        else
        {
            pixels[pixel] = packRGBA({ 0, 0, 0, 1 });
        }
    }
}

extern "C" __global__ void render(float4* accumulators, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* attribs, float3 p_light, int iteration)
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if ( imageSize.x <= xi || imageSize.y <= yi )
    {
        return;
    }

    int pixel = xi + yi * imageSize.x;

    int dimLevel = 0;
    float2 jitter;
    sobol::shuffled_scrambled_sobol_2d(&jitter.x, &jitter.y, iteration, xi, yi, dimLevel++);

    float3 ro, rd;
    rayGenerator.shoot(&ro, &rd, (float)(xi + jitter.x) / imageSize.x, (float)(yi + jitter.y) / imageSize.y);

    Hit hit;
    intersect_stackfree(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd));
    if (hit.t == MINIMUM_LBVH_FLT_MAX)
    {
        accumulators[pixel] += {0.0f, 0.0f, 0.0f, 1.0f};
        return;
    }

    float3 p = ro + rd * hit.t;
    float3 n = normalize(hit.ng);

    if (0.0f < dot(n, rd))
    {
        n = -n;
    }

    if (attribs[hit.triangleIndex].material != Material::Diffuse)
    {
        // handle later
        accumulators[pixel] += {0.0f, 1.0f, 1.0f, 1.0f};
        return;
    }

    float3 toLight = p_light - p;
    float d2 = dot(toLight, toLight);
    float3 reflectance = { 0.75f, 0.75f, 0.75f };

    float3 L = {};

    bool invisible = occluded(internals, triangles, *rootNode, p, n, p_light, { 0, 0, 0 });
    if (!invisible)
    {
        float3 light_intencity = { 1, 1, 1 };
        L += reflectance * light_intencity / d2 * fmaxf(dot(normalize(toLight), n), 0.0f);
    }

    accumulators[pixel] += {L.x, L.y, L.z, 1.0f};
}

extern "C" __global__ void pack( uint32_t* pixels, float4* accumulators, int n )
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    if (n <= xi)
    {
        return;
    }
    float4 acc = accumulators[xi];
    pixels[xi] = packRGBA({
        powf(acc.x / acc.w, 1.0f / 2.2f),
        powf(acc.y / acc.w, 1.0f / 2.2f),
        powf(acc.z / acc.w, 1.0f / 2.2f),
        1.0f }
    );
}