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
    return 0xFF000000 | (i4.z << 16) | (i4.y << 8) | i4.x;
}

__device__ inline float3 viridis(float t)
{
    t = clamp(t, 0.0f, 1.0f);
    float3 c0 = make_float3(0.2777273272234177f, 0.005407344544966578f, 0.3340998053353061f);
    float3 c1 = make_float3(0.1050930431085774f, 1.404613529898575f, 1.384590162594685f);
    float3 c2 = make_float3(-0.3308618287255563f, 0.214847559468213f, 0.09509516302823659f);
    float3 c3 = make_float3(-4.634230498983486f, -5.799100973351585f, -19.33244095627987f);
    float3 c4 = make_float3(6.228269936347081f, 14.17993336680509f, 56.69055260068105f);
    float3 c5 = make_float3(4.776384997670288f, -13.74514537774601f, -65.35303263337234f);
    float3 c6 = make_float3(-5.435455855934631f, 4.645852612178535f, 26.3124352495832f);

    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
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
extern "C" __global__ void __launch_bounds__(32) photonTrace(const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* attribs, float3 p_light, float eta_min, float eta_max, int iteration, PathCache pathCache, float minThroughput, float3* debugPoints, int* debugPointCount, NodeIndex* stackBuffer)
{
    StackBufferWarp stackBufferWarp(stackBuffer);

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

    enum {
        K = PHOTON_TRACE_MAX_DEPTH
    };

    for (int j = 0; j < 8; j++)
    {
        float2 params = {};
        sobol::shuffled_scrambled_sobol_2d(&params.x, &params.y, j * blockDim.x + threadIdx.x, iteration, iTri, 789);
        params = square2triangle(params);

        float2 eta_random;
        sobol::shuffled_scrambled_sobol_2d(&eta_random.x, &eta_random.y, j * blockDim.x + threadIdx.x, iteration, iTri, 178);
        float eta = lerp(eta_min, eta_max, eta_random.x);

        float3 e0 = tri.vs[1] - tri.vs[0];
        float3 e1 = tri.vs[2] - tri.vs[0];
        float3 p = tri.vs[0] + e0 * params.x + e1 * params.y;

        float3 ro = p_light;
        float3 rd = p - p_light;

        EventDescriptor eDescriptor;
        bool admissiblePath = false;
        int triIndices[K];
        float parameters[K * 2];
        float3 p_final;
        float throughtput = 1.0f;
        for (int d = 0; d < K + 1; d++)
        {
            minimum_lbvh::Hit hit;
            minimum_lbvh::intersect_stackfull(&hit, internals, triangles, *rootNode, ro, rd, minimum_lbvh::invRd(rd), stackBufferWarp.stack());
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

            if (m == Material::Diffuse)
            {
                if (d == 0)
                {
                    break;
                }

                admissiblePath = true;
                p_final = p_hit;
                break;
            }
            if (d == K) // finish
            {
                break;
            }

            float3 wi = -rd;
            triIndices[d] = hit.triangleIndex;
            parameters[d * 2] = hit.uv.x;
            parameters[d * 2 + 1] = hit.uv.y;

            float2 random;
            sobol::shuffled_scrambled_sobol_2d(&random.x, &random.y, j * blockDim.x + threadIdx.x, d, iTri, iteration);

            float reflectance = fresnel_exact_norm_free(wi, ns, eta);

            Event e;
            if (m == Material::Mirror)
            {
                e = Event::R;
            }
            else if (m == Material::Dielectric)
            {
                if (random.x < reflectance)
                {
                    e = Event::R;
                }
                else
                {
                    e = Event::T;
                }
            }

            eDescriptor.push_back(e);

            float sign = 1.0f;
            float3 wo;
            if (e == Event::R)
            {
                wo = reflection(wi, ns);

                if (m == Material::Dielectric)
                {
                    throughtput *= reflectance;
                }
            }
            else if (e == Event::T)
            {
                if (refraction_norm_free(&wo, wi, ns, eta) == false)
                {
                    break;
                }

                sign = -1.0f;
                throughtput *= 1.0f - reflectance;
            }

            if (0.0f < dot(hit.ng, wi) * dot(hit.ng, wo) * sign) // geometrically admissible
            {
                float3 offset_dir = normalize(hit.ng * dot(hit.ng, wo));
                ro = p_hit + offset_dir * rayOffsetScale(p_hit);
                rd = wo;
                continue;
            }

            break;
        }
        if (admissiblePath && minThroughput < throughtput)
        {
            bool success = pathCache.store(p_final, triIndices, parameters, eDescriptor);

#if defined(SHOW_VALID_CACHE)
            if (success)
            {
                int index = atomicAdd(debugPointCount, 1);
                debugPoints[index] = p_final;
            }
#endif
        }
    }
}

extern "C" __global__ void __launch_bounds__(16 * 16) solvePrimary(float4* accumulators, FirstDiffuse* firstDiffuses, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* triangleAttribs, float3 p_light, float lightIntencity, float radianceClamp, CauchyDispersion cauchy, Texture8RGBX floorTex, int iteration, NodeIndex *stackBuffer)
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    StackBufferWarp stackBufferWarp(stackBuffer);

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

    float2 lambda_random;
    sobol::shuffled_scrambled_sobol_2d(&lambda_random.x, &lambda_random.y, iteration, xi, yi, dimLevel++);
    float lambda = CIE_2015_10deg::cmf_y_sample(lambda_random.x);
    float p_lambda = CIE_2015_10deg::cmf_y_pdf(lambda);
    float eta = cauchy(lambda);

    bool hasDiffuseHit = false;
    minimum_lbvh::Hit hit_last;
    for (int d = 0; d < 32; d++)
    {
        minimum_lbvh::Hit hit;
        minimum_lbvh::intersect_stackfull(&hit, internals, triangles, *rootNode, ro, rd, minimum_lbvh::invRd(rd), stackBufferWarp.stack());
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

        float sign = 1.0f;
        float3 wo;
        if (e == Event::R)
        {
            wo = reflection(wi, ns);
        }
        else
        {
            if (refraction_norm_free(&wo, wi, ns, eta) == false)
            {
                break;
            }
            sign = -1.0f;
        }

        if (0.0f < dot(hit.ng, wi) * dot(hit.ng, wo) * sign) // geometrically admissible
        {
            float3 offset_dir = normalize(hit.ng * dot(hit.ng, wo));
            ro = p_hit + offset_dir * rayOffsetScale(p_hit);
            rd = wo;
            continue;
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

#if 0
    float scale = 10.0f;
    int x = floor(p.x * scale);
    int z = floor(p.z * scale);
    float chess = (x + z) % 2;
    float3 reflectance = lerp(float3{ 0.5f, 0.5f, 0.5f }, float3{ 0.75f, 0.75f, 0.75f }, chess);
#else
    float3 reflectance = floorTex.samplePoint(p.x, p.z);
#endif

    float3 L = {};

    bool invisible = occluded(internals, triangles, *rootNode, p, n, p_light, { 0, 0, 0 });
    if (!invisible)
    {
        //L += reflectance * lightIntencity / d2 * fmaxf(dot(normalize(toLight), n), 0.0f);
        float contrib = lightIntencity / d2 * fmaxf(dot(normalize(toLight), n), 0.0f);

        contrib = fminf(contrib, radianceClamp); // reason of variance. need to think

        float3 xyz = {
            CIE_2015_10deg::cmf_x(lambda) / INTEGRAL_OF_CMF_Y_IN_NM,
            CIE_2015_10deg::cmf_y(lambda) / INTEGRAL_OF_CMF_Y_IN_NM,
            CIE_2015_10deg::cmf_z(lambda) / INTEGRAL_OF_CMF_Y_IN_NM,
        };
        float3 srgblinear = xyz2srgblinear(xyz);
        L += reflectance /* diffuse reflectance */ * srgblinear * contrib / p_lambda;
    }

    firstDiffuses[pixel].p = p;
    firstDiffuses[pixel].ng = n;
    firstDiffuses[pixel].R = reflectance;
    firstDiffuses[pixel].lambda = lambda;

    accumulators[pixel] += {L.x, L.y, L.z, 1.0f};
}

extern "C" __global__ void lookupFlatten(EventDescriptor eDescriptor, SpecularPath* specularPaths, uint32_t *specularPathCounter, const FirstDiffuse* firstDiffuses, int2 imageSize, PathCache pathCache )
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
    int nPaths = 0;
    pathCache.lookUpIndex(p, eDescriptor, [&](int index) {
        nPaths++;
    });

    //atomicAdd(specularPathCounter, nPaths);
    uint32_t head = atomicAdd(specularPathCounter, nPaths);

    if (MAX_SPECULAR_PATH_COUNT <= head + nPaths)
    {
        // should not happen..
        return;
    }

    int iPath = 0;
    pathCache.lookUpIndex(p, eDescriptor, [&](int index) {
        specularPaths[head + iPath].pixel = pixel;
        specularPaths[head + iPath].cacheIndex = index;

        iPath++;
    });
}

template <int K>
__device__ void solveSpecularPath(float4* accumulators, SpecularPath* specularPaths, uint32_t specularPathCount, const FirstDiffuse* firstDiffuses, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* triangleAttribs, float3 p_light, float lightIntencity, PathCache* pathCache, EventDescriptor eDescriptor, CauchyDispersion cauchy, int iteration)
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;

    if (specularPathCount <= xi)
    {
        return;
    }

    SpecularPath specularPath = specularPaths[xi];
    FirstDiffuse firstDiffuse = firstDiffuses[specularPath.pixel];

    float3 p = firstDiffuse.p;
    float3 n = firstDiffuse.ng;
    float3 R = firstDiffuse.R;
    float lambda = firstDiffuse.lambda;

    const PathCache::TrianglePath& thePath = pathCache->m_pathes[specularPath.cacheIndex];

    minimum_lbvh::Triangle tris[K];
    TriangleAttrib attribs[K];
    for (int k = 0; k < K; k++)
    {
        int indexOfTri = thePath.tris[k];
        tris[k] = triangles[indexOfTri];
        attribs[k] = triangleAttribs[indexOfTri];
    }
    float parameters[K * 2];
    for (int i = 0; i < K * 2; i++)
    {
        parameters[i] = thePath.parameters[i];
        //parameters[i] = 1.0f / 3.0f;
    }

    float p_lambda = CIE_2015_10deg::cmf_y_pdf(lambda);

    float eta = cauchy(lambda);
    bool converged = solveConstraints<K>(parameters, p_light, p, tris, attribs, eta, eDescriptor, 32, 1.0e-10f, 1 /* warm up */);

    if (converged)
    {
        float throughput = contributableThroughput<K>(
            parameters, p_light, p, tris, attribs, eDescriptor,
            internals, triangles, *rootNode, eta);

        if (0.0f < throughput)
        {
            float dAdwValue = dAdw(p_light, getVertex(0, tris, parameters) - p_light, p, tris, attribs, eDescriptor, eta);
            // L += throughput * R * lightIntencity / dAdwValue * fmaxf(dot(normalize(getVertex(K - 1, tris, parameters) - p), n), 0.0f);

            float contrib = throughput * lightIntencity / dAdwValue * fmaxf(dot(normalize(getVertex(K - 1, tris, parameters) - p), n), 0.0f);

            float3 xyz = {
                CIE_2015_10deg::cmf_x(lambda) / INTEGRAL_OF_CMF_Y_IN_NM,
                CIE_2015_10deg::cmf_y(lambda) / INTEGRAL_OF_CMF_Y_IN_NM,
                CIE_2015_10deg::cmf_z(lambda) / INTEGRAL_OF_CMF_Y_IN_NM,
            };
            float3 srgblinear = xyz2srgblinear(xyz);
            float3 L = R /* diffuse reflectance */ * srgblinear * contrib / p_lambda;

            atomicAdd(&accumulators[specularPath.pixel].x, L.x);
            atomicAdd(&accumulators[specularPath.pixel].y, L.y);
            atomicAdd(&accumulators[specularPath.pixel].z, L.z);
        }
    }
}

#define DECL_SOLVE_SPECULAR_PATH( k ) \
extern "C" __global__ void __launch_bounds__(256) solveSpecularPath_K##k(float4* accumulators, SpecularPath* specularPaths, uint32_t specularPathCount, const FirstDiffuse* firstDiffuses, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, const TriangleAttrib* triangleAttribs, float3 p_light, float lightIntencity, PathCache pathCache, EventDescriptor eDescriptor, CauchyDispersion cauchy, int iteration) \
{\
    solveSpecularPath<k>(accumulators, specularPaths, specularPathCount, firstDiffuses, rootNode, internals, triangles, triangleAttribs, p_light, lightIntencity, &pathCache, eDescriptor, cauchy, iteration); \
}

DECL_SOLVE_SPECULAR_PATH(1);
DECL_SOLVE_SPECULAR_PATH(2);
DECL_SOLVE_SPECULAR_PATH(3);
DECL_SOLVE_SPECULAR_PATH(4);

extern "C" __global__ void pack( uint32_t* pixels, float4* accumulators, int n )
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    if (n <= xi)
    {
        return;
    }
    float4 acc = accumulators[xi];
    pixels[xi] = packRGBA({
        srgb_oetf(acc.x / acc.w),
        srgb_oetf(acc.y / acc.w),
        srgb_oetf(acc.z / acc.w),
        1.0f }
    );
}