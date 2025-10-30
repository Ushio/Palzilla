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


extern "C" __global__ void __launch_bounds__(16 * 16) solvePrimary(float4* accumulators, FirstDiffuse* firstDiffuses, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* triangleAttribs, float3 p_light, float eta, int iteration)
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (imageSize.x <= xi || imageSize.y <= yi)
    {
        return;
    }

    int pixel = xi + yi * imageSize.x;

    int dimLevel = 0;
    float2 jitter;
    sobol::shuffled_scrambled_sobol_2d(&jitter.x, &jitter.y, iteration, xi, yi, dimLevel++);

    float3 ro, rd;
    rayGenerator.shoot(&ro, &rd, (float)(xi + jitter.x) / imageSize.x, (float)(yi + jitter.y) / imageSize.y);

    bool hasDiffuseHit = false;
    minimum_lbvh::Hit hit_last;
    for (int d = 0; d < 16; d++)
    {
        minimum_lbvh::Hit hit;
        minimum_lbvh::intersect_stackfree(&hit, internals, triangles, *rootNode, ro, rd, minimum_lbvh::invRd(rd));
        if (hit.t == MINIMUM_LBVH_FLT_MAX)
        {
            hit_last = hit;
            break;
        }
        TriangleAttrib attrib = triangleAttribs[hit.triangleIndex];
        Material m = attrib.material;

        float2 random;
        sobol::shuffled_scrambled_sobol_2d(&random.x, &random.y, iteration, xi, yi, dimLevel++);

        if (m == Material::Diffuse)
        {
            hasDiffuseHit = true;
            hit_last = hit;
            break;
        }

        float3 p_hit = ro + rd * hit.t;

        float3 ns =
            attrib.shadingNormals[0] +
            (attrib.shadingNormals[1] - attrib.shadingNormals[0]) * hit.uv.x +
            (attrib.shadingNormals[2] - attrib.shadingNormals[0]) * hit.uv.y;
        float3 ng = dot(ns, hit.ng) < 0.0f ? -hit.ng : hit.ng; // aligned
        float3 wi = -rd;

        Event e;
        if (m == Material::Mirror)
        {
            e = Event::R;
        }
        else if (m == Material::Dielectric)
        {
            float reflectance = fresnel_exact_norm_free(wi, ns, eta);
            if (random.x < reflectance)
            {
                e = Event::R;
            }
            else
            {
                e = Event::T;
            }
        }

        if (e == Event::R)
        {
            float3 wo = reflection(wi, ns);

            if (0.0f < dot(ng, wi) * dot(ng, wo)) // geometrically admissible
            {
                float3 ng_norm = normalize(ng);
                ro = p_hit + (dot(wo, ng) < 0.0f ? -ng_norm : ng_norm) * rayOffsetScale(p_hit);
                rd = wo;
                continue;
            }
        }
        else
        {
            float3 wo;
            if (refraction_norm_free(&wo, wi, ns, eta) == false)
            {
                break;
            }

            if (dot(ng, wi) * dot(ng, wo) < 0.0f) // geometrically admissible
            {
                float3 ng_norm = normalize(ng);
                ro = p_hit + (dot(wo, ng) < 0.0f ? -ng_norm : ng_norm) * rayOffsetScale(p_hit);
                rd = wo;
                continue;
            }
        }
        break;
    }

    if ( hasDiffuseHit == false )
    {
        accumulators[pixel] += {0.0f, 0.0f, 0.0f, 1.0f};
        firstDiffuses[pixel].R = { 0.0f, 0.0f, 0.0f };
        return;
    }

    float3 p = ro + rd * hit_last.t;
    float3 n = normalize(hit_last.ng);

    if (0.0f < dot(n, rd))
    {
        n = -n;
    }

    float3 toLight = p_light - p;
    float d2 = dot(toLight, toLight);
    float3 reflectance = { 0.75f, 0.75f, 0.75f };

    float3 L = {};
    float3 light_intencity = { 1, 1, 1 };

    bool invisible = occluded(internals, triangles, *rootNode, p, n, p_light, { 0, 0, 0 });
    if (!invisible)
    {
        L += reflectance * light_intencity / d2 * fmaxf(dot(normalize(toLight), n), 0.0f);
    }

    firstDiffuses[pixel].p = p;
    firstDiffuses[pixel].ng = n;
    firstDiffuses[pixel].R = reflectance;

    accumulators[pixel] += {L.x, L.y, L.z, 1.0f};
}

template <int K>
__device__ void solveSpecular(float4* accumulators, const FirstDiffuse* firstDiffuses, int2 imageSize, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* triangleAttribs, float3 p_light, PathCache* pathCache, EventDescriptor eDescriptor, float eta, int iteration)
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (imageSize.x <= xi || imageSize.y <= yi)
    {
        return;
    }

    int pixel = xi + yi * imageSize.x;
    FirstDiffuse firstDiffuse = firstDiffuses[pixel];

    float3 p = firstDiffuse.p;
    float3 n = firstDiffuse.ng;
    float3 R = firstDiffuse.R;

    float3 toLight = p_light - p;
    float d2 = dot(toLight, toLight);

    float3 L = {};
    float3 light_intencity = { 1, 1, 1 };

    pathCache->lookUp(p, [&](const int triIndices[], const float photon_parameters[]) {
        minimum_lbvh::Triangle tris[K];
        TriangleAttrib attribs[K];
        for (int k = 0; k < K; k++)
        {
            int indexOfTri = triIndices[k];
            tris[k] = triangles[indexOfTri];
            attribs[k] = triangleAttribs[indexOfTri];
        }
        float parameters[K * 2];
        for (int i = 0; i < K * 2; i++)
        {
            parameters[i] = photon_parameters[i];
            //parameters[i] = 1.0f / 3.0f;
        }
        bool converged = solveConstraints<K>(parameters, p_light, p, tris, attribs, eta, eDescriptor, 32, 1.0e-10f);

        if (converged)
        {
            float throughput = contributableThroughput<K>(
                parameters, p_light, p, tris, attribs, eDescriptor,
                internals, triangles, *rootNode, eta);

            if (0.0f < throughput)
            {
                float dAdwValue = dAdw(p_light, getVertex(0, tris, parameters) - p_light, p, tris, attribs, eDescriptor, K, eta);
                L += throughput * R * light_intencity / dAdwValue * fmaxf(dot(normalize(getVertex(K - 1, tris, parameters) - p), n), 0.0f);
            }
        }
        });

    accumulators[pixel] += {L.x, L.y, L.z, 0.0f};
}

#define DECL_SOLVE_SPECULAR_TRACE( k ) \
extern "C" __global__ void __launch_bounds__(16 * 16) solveSpecular_K##k(float4* accumulators, const FirstDiffuse* firstDiffuses, int2 imageSize, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* triangleAttribs, float3 p_light, PathCache pathCache, EventDescriptor eDescriptor, float eta, int iteration) \
{\
    solveSpecular<k>(accumulators, firstDiffuses, imageSize, rootNode, internals, triangles, triangleAttribs, p_light, &pathCache, eDescriptor, eta, iteration);\
}

DECL_SOLVE_SPECULAR_TRACE(1);
DECL_SOLVE_SPECULAR_TRACE(2);
DECL_SOLVE_SPECULAR_TRACE(3);
DECL_SOLVE_SPECULAR_TRACE(4);
DECL_SOLVE_SPECULAR_TRACE(5);
DECL_SOLVE_SPECULAR_TRACE(6);
DECL_SOLVE_SPECULAR_TRACE(7);


template <int K>
__device__ void photonTrace(const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* attribs, float3 p_light, EventDescriptor eDescriptor, float eta, int iteration, PathCache* pathCache, float minThroughput, float3* debugPoints, int* debugPointCount)
{
    int iTri = blockIdx.x;
    if (attribs[iTri].material == Material::Diffuse)
    {
        return;
    }

    minimum_lbvh::Triangle tri = triangles[iTri];

    // Skip backface
    float3 ng = minimum_lbvh::unnormalizedNormalOf(tri);
    float3 ns = attribs[iTri].shadingNormals[0];
    ng *= dot(ng, ns);
    if (dot(ng, p_light - tri.vs[0]) < 0.0f)
    {
        return;
    }

    int contiguousFails = 0;
    for (int j = 0; ; j++)
    {
        float2 params = {};
        sobol::shuffled_scrambled_sobol_2d(&params.x, &params.y, j * blockDim.x + threadIdx.x, iteration, iTri, 789);
        params = square2triangle(params);

        float3 e0 = tri.vs[1] - tri.vs[0];
        float3 e1 = tri.vs[2] - tri.vs[0];
        float3 p = tri.vs[0] + e0 * params.x + e1 * params.y;

        float3 ro = p_light;
        float3 rd = p - p_light;

        bool admissiblePath = false;
        int triIndices[K];
        float parameters[K * 2];
        float3 p_final;
        float throughtput = 1.0f;
        for (int d = 0; d < K + 1; d++)
        {
            minimum_lbvh::Hit hit;
            minimum_lbvh::intersect_stackfree(&hit, internals, triangles, *rootNode, ro, rd, minimum_lbvh::invRd(rd));
            if (hit.t == MINIMUM_LBVH_FLT_MAX)
            {
                break;
            }
            TriangleAttrib attrib = attribs[hit.triangleIndex];
            Material m = attrib.material;
            float3 p_hit = ro + rd * hit.t;

            float3 ns =
                attrib.shadingNormals[0] +
                (attrib.shadingNormals[1] - attrib.shadingNormals[0]) * hit.uv.x +
                (attrib.shadingNormals[2] - attrib.shadingNormals[0]) * hit.uv.y;
            float3 ng = dot(ns, hit.ng) < 0.0f ? -hit.ng : hit.ng; // aligned

            if (d == K)
            {
                if (m == Material::Diffuse)
                {
                    // store 
                    admissiblePath = true;
                    p_final = p_hit;
                }
                break;
            }

            float3 wi = -rd;
            triIndices[d] = hit.triangleIndex;
            parameters[d * 2] = hit.uv.x;
            parameters[d * 2 + 1] = hit.uv.y;

            if (eDescriptor.get(d) == Event::R && (m == Material::Mirror || m == Material::Dielectric))
            {
                float3 wo = reflection(wi, ns);

                if (0.0f < dot(ng, wi) * dot(ng, wo)) // geometrically admissible
                {
                    float3 ng_norm = normalize(ng);
                    ro = p_hit + (dot(wo, ng) < 0.0f ? -ng_norm : ng_norm) * rayOffsetScale(p_hit);
                    rd = wo;

                    if (m == Material::Dielectric)
                    {
                        throughtput *= fresnel_exact_norm_free(wi, ns, eta);
                    }
                    continue;
                }
            }
            if (eDescriptor.get(d) == Event::T && m == Material::Dielectric)
            {
                float3 wo;

                if (refraction_norm_free(&wo, wi, ns, eta) == false)
                {
                    break;
                }

                if (dot(ng, wi) * dot(ng, wo) < 0.0f) // geometrically admissible
                {
                    float3 ng_norm = normalize(ng);
                    ro = p_hit + (dot(wo, ng) < 0.0f ? -ng_norm : ng_norm) * rayOffsetScale(p_hit);
                    rd = wo;

                    throughtput *= 1.0f - fresnel_exact_norm_free(wi, ns, eta);
                    continue;
                }
            }
            break;
        }
        bool success = false; 
        if (admissiblePath && minThroughput < throughtput )
        {
            success = pathCache->store(p_final, triIndices, parameters, K);

            //if (success)
            //{
            //    int index = atomicAdd(debugPointCount, 1);
            //    debugPoints[index] = p_final;
            //}
        }

        if (__any(success))
        {
            contiguousFails = 0;
        }
        else
        {
            contiguousFails++;
        }
        if (2 <= contiguousFails)
        {
            break;
        }
    }
}

#define DECL_PHOTON_TRACE( k ) \
extern "C" __global__ void __launch_bounds__(32) photonTrace_K##k(const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* attribs, float3 p_light, EventDescriptor eDescriptor, float eta, int iteration, PathCache pathCache, float minThroughput, float3* debugPoints, int* debugPointCount)\
{\
    photonTrace<k>(rootNode, internals, triangles, attribs, p_light, eDescriptor, eta, iteration, &pathCache, minThroughput, debugPoints, debugPointCount);\
}

DECL_PHOTON_TRACE(1)
DECL_PHOTON_TRACE(2)
DECL_PHOTON_TRACE(3)
DECL_PHOTON_TRACE(4)
DECL_PHOTON_TRACE(5)
DECL_PHOTON_TRACE(6)
DECL_PHOTON_TRACE(7)

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