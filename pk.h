#pragma once
#include "helper_math.h"

#define ENABLE_GPU_BUILDER
#include "minimum_lbvh.h"
#include "sen.h"
#include "saka.h"
#include "typedbuffer.h"

//#define ENABLE_PATH_CUTS 

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define PK_KERNELCC
#define PK_DEVICE __device__
#else
#include <initializer_list>
#include <memory>
#include "pr.hpp"
#include "Orochi/Orochi.h"
#include "shader.h"
#include "tinyhiponesweep.h"
#include "camera.h"
#define PK_DEVICE
#endif

#define MIN_VERTEX_DIST 1.0e-5f

PK_DEVICE inline float rayOffsetScale(float3 p)
{
    p = fabs(p);
    float maxElem = fmaxf(fmaxf(fmaxf(p.x, p.y), p.z), 1.0f);
    float flt_eps = 1.192092896e-07F;
    return maxElem * flt_eps * 16.0f;
}

enum class Material : int
{
    Diffuse,
    Mirror,
    Dielectric,
};

struct TriangleAttrib
{
    Material material;
    float3 shadingNormals[3];
};

struct FirstDiffuse
{
    float3 p;
    float3 ng;
    float3 R;
};

PK_DEVICE inline bool occluded(
    const minimum_lbvh::InternalNode* nodes,
    const minimum_lbvh::Triangle* triangles,
    minimum_lbvh::NodeIndex node,
    float3 from,
    float3 from_n,
    float3 to,
    float3 to_n)
{
    if (dot(to - from, from_n) < 0.0f)
    {
        from_n = -from_n;
    }
    if (dot(from - to, to_n) < 0.0f)
    {
        to_n = -to_n;
    }

    float3 from_safe = from + from_n * rayOffsetScale(from);
    float3 to_safe = to + to_n * rayOffsetScale(to);

    float3 rd = to_safe - from_safe;

    minimum_lbvh::Hit hit;
    hit.t = 1.0f;
    minimum_lbvh::intersect_stackfree(&hit, nodes, triangles, node, from_safe, rd, minimum_lbvh::invRd(rd), minimum_lbvh::RAY_QUERY_ANY);
    return hit.t < 1.0f;
}

enum class Event
{
    R = 0,
    T = 1
};
struct EventDescriptor
{
    PK_DEVICE EventDescriptor() : m_events(0) {}

#if !defined(PK_KERNELCC)
    EventDescriptor(std::initializer_list<Event> events) : m_events(0)
    {
        int index = 0;
        for (Event e : events)
        {
            this->set(index++, e);
        }
    }
#endif

    PK_DEVICE Event get(uint32_t index) const {
        bool setbit = m_events & (1u << index);
        return setbit ? Event::T : Event::R;
    }
    PK_DEVICE void set(uint32_t index, Event e)
    {
        m_events &= ~(1u << index);
        if (e == Event::T)
        {
            m_events |= 1u << index;
        }
    }
    uint32_t m_events;
};

// ALow-DistortionMapBetweenTriangleandSquare
PK_DEVICE inline float2 square2triangle(float2 square)
{
    if (square.y > square.x)
    {
        square.x *= 0.5f;
        square.y -= square.x;
    }
    else
    {
        square.y *= 0.5f;
        square.x -= square.y;
    }
    return square;
}

// normal dir defines in-out of the medium
PK_DEVICE inline bool refraction_norm_free(float3* wo, float3 wi, float3 n, float eta /* = eta_t / eta_i */)
{
    float NoN = dot(n, n);
    float WIoN = dot(wi, n);
    if (WIoN < 0.0f)
    {
        WIoN = -WIoN;
        n = -n;
        eta = 1.0f / eta;
    }

    float WIoWI = dot(wi, wi);
    float k = NoN * WIoWI * (eta * eta - 1.0f) + WIoN * WIoN;
    if (k < 0.0f)
    {
        return false;
    }
    *wo = -wi * NoN + (WIoN - sqrtf(k)) * n;
    return true;
}

PK_DEVICE inline float3 reflection(float3 wi, float3 n)
{
    return n * dot(wi, n) * 2.0f / dot(n, n) - wi;
}

PK_DEVICE inline uint32_t hash_combine(uint32_t seed, uint32_t h)
{
    return h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
PK_DEVICE inline uint32_t hash_of_iP(int x, int y, int z)
{
    uint32_t h = 12345;
    h = hash_combine(h, x * 2654435761);
    h = hash_combine(h, y * 805459861);
    h = hash_combine(h, z * 3674653429);
    return h | 1u;
}
PK_DEVICE inline uint32_t spacial_hash(float3 p, float spacial_step) {
    float3 indexf = (p / spacial_step);
    int x = floorf(indexf.x);
    int y = floorf(indexf.y);
    int z = floorf(indexf.z);
    return hash_of_iP(x, y, z);
}

struct PathCache
{
    enum {
        MAX_PATH_LENGTH = 10,
        CACHE_STORAGE_COUNT = 1u << 22
    };

    struct TrianglePath
    {
        uint32_t hashOfP;
        int tris[MAX_PATH_LENGTH];
        float parameters[MAX_PATH_LENGTH * 2];
    };

#if !defined(PK_KERNELCC)
    PathCache(TYPED_BUFFER_TYPE type): m_hashsOfPath(type), m_pathes(type), m_numberOfCached(type)
    {
    }
    void init( float step )
    {
        m_spatial_step = step;
        m_hashsOfPath.allocate(CACHE_STORAGE_COUNT);
        m_pathes.allocate(CACHE_STORAGE_COUNT);
        m_numberOfCached.allocate(1);
        clear();
    }
    void clear()
    {
        if (m_hashsOfPath.isHost())
        {
            for (int i = 0; i < m_hashsOfPath.size(); i++)
            {
                m_hashsOfPath[i] = 0;
            }
            m_numberOfCached[0] = 0;
        }
        else
        {
            oroMemsetD32(m_numberOfCached.data(), 0, 1);
            oroMemsetD32(m_hashsOfPath.data(), 0, m_hashsOfPath.size());
        }
    }
    uint32_t atomicCAS(uint32_t* address, uint32_t compare, uint32_t val)
    {
        return InterlockedCompareExchange((volatile LONG *)address, (LONG)val, (LONG)compare);
    }
    void atomicInc(uint32_t* address, uint32_t)
    {
        InterlockedIncrement((volatile LONG*)address);
    }
    float occupancy() const
    {
        if (m_numberOfCached.isDevice())
        {
            uint32_t numberOfCached[1];
            numberOfCached << m_numberOfCached;
            return (float)numberOfCached[0] / CACHE_STORAGE_COUNT;
        }
        return (float)m_numberOfCached[0] / CACHE_STORAGE_COUNT;
    }
#endif

    // return true when store was succeeded
    PK_DEVICE bool store(float3 pos, int tris[], float* parameters, int K)
    {
        uint32_t hashOfPath = 123;
        for (int d = 0; d < K; d++)
        {
            hashOfPath = minimum_lbvh::hashPCG(hashOfPath + tris[d]);
        }
        hashOfPath |= 1u;

        bool success = false;

        float3 indexf = pos / m_spatial_step;
        int x = floorf(indexf.x);
        int y = floorf(indexf.y);
        int z = floorf(indexf.z);
        int dx = indexf.x - x < 0.5f ? -1 : 1;
        int dy = indexf.y - y < 0.5f ? -1 : 1;
        int dz = indexf.z - z < 0.5f ? -1 : 1;
        for (int iz = 0; iz < 2; iz++)
        for (int iy = 0; iy < 2; iy++)
        for (int ix = 0; ix < 2; ix++)
        {
            uint32_t hashOfP = hash_of_iP(x + ix * dx, y + iy * dy, z + iz * dz);
            uint32_t home = hashOfP % CACHE_STORAGE_COUNT;
            for (int offset = 0; offset < CACHE_STORAGE_COUNT; offset++)
            {
                uint32_t index = (home + offset) % CACHE_STORAGE_COUNT;
                uint32_t old = atomicCAS(&m_hashsOfPath[index], 0, hashOfPath);

                if (old == 0) // inserted
                {
                    m_hashsOfPath[index] = hashOfPath;
                    m_pathes[index].hashOfP = hashOfP;
                    for (int d = 0; d < K; d++)
                    {
                        m_pathes[index].tris[d] = tris[d];
                    }
                    if (parameters)
                    {
                        for (int i = 0; i < K * 2; i++)
                        {
                            m_pathes[index].parameters[i] = parameters[i];
                        }
                    }

                    success = true;
                    atomicInc(&m_numberOfCached[0], 0xFFFFFFFF);
                    break;
                }
                else if (old == hashOfPath) // existing
                {
                    break;
                }
            }
        }

        return success;
    }

    template <class F>
    PK_DEVICE void lookUp(float3 p, F f) const
    {
        uint32_t hashOfP = spacial_hash(p, m_spatial_step);
        uint32_t home = hashOfP % CACHE_STORAGE_COUNT;
        for (int offset = 0; offset < CACHE_STORAGE_COUNT; offset++)
        {
            uint32_t index = (home + offset) % CACHE_STORAGE_COUNT;
            if (m_hashsOfPath[index] == 0)
            {
                break; // no more cached
            }
            if (m_pathes[index].hashOfP != hashOfP)
            {
                continue;
            }
            f(m_pathes[index].tris, m_pathes[index].parameters);
        }
    }

    float m_spatial_step = 1.0f;
    TypedBuffer<uint32_t> m_hashsOfPath;
    TypedBuffer<TrianglePath> m_pathes;
    TypedBuffer<uint32_t> m_numberOfCached;
};

// normal dir defines in-out of the medium
PK_DEVICE inline float fresnel_exact(float3 wi, float3 n, float eta /* eta_t / eta_i */) {
    float c = dot(n, wi);
    if (c < 0.0f)
    {
        c = -c;
        n = -n;
        eta = 1.0f / eta;
    }
    float k = eta * eta - 1.0f + c * c;
    if (k < 0.0f)
    {
        return 1.0f; // TIR
    }
    float g = sqrtf(k);

    auto sqr = [](float x) { return x * x; };
    float gmc = g - c;
    float gpc = g + c;
    return 0.5f * sqr(gmc / gpc) * (1.0f + sqr((c * gpc - 1.0f) / (c * gmc + 1.0f)));
}

// normal dir defines in-out of the medium
PK_DEVICE inline float fresnel_exact_norm_free(float3 wi, float3 n, float eta /* eta_t / eta_i */) {
    float nn = dot(n, n);
    float wiwi = dot(wi, wi);
    float C = dot(n, wi);

    if( C < 0.0f )
    {
        C = -C;
        n = -n;
        eta = 1.0f / eta;
    }

    float K = nn * wiwi * (eta * eta - 1.0f) + C * C;
    if (K < 0.0f)
    {
        return 1.0f; // TIR
    }
    float G = sqrtf(K);

    auto sqr = [](float x) { return x * x; };
    float GmC = G - C;
    float GpC = G + C;
    return 0.5f * sqr(GmC / GpC) * (1.0f + sqr((C * GpC - nn * wiwi) / (C * GmC + nn * wiwi)));
}

// Building an Orthonormal Basis, Revisited
PK_DEVICE void GetOrthonormalBasis(float3 zaxis, float3* xaxis, float3* yaxis) {
    const float sign = copysignf(1.0f, zaxis.z);
    const float a = -1.0f / (sign + zaxis.z);
    const float b = zaxis.x * zaxis.y * a;
    *xaxis = { 1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x };
    *yaxis = { b, sign + zaxis.y * zaxis.y * a, -zaxis.y };
}

struct SolverEmptyCallback {
    PK_DEVICE void operator()( int iter, bool converged ) const{}
};

// parameters: output barycentric coordinates
template <int K, class callback = SolverEmptyCallback >
PK_DEVICE inline bool solveConstraints(float parameters[K * 2], float3 p_beg, float3 p_end, minimum_lbvh::Triangle tris[K], TriangleAttrib attribs[K], float eta, EventDescriptor eDescriptor, int maxIterations, float costTolerance, callback end_of_iter = SolverEmptyCallback())
{
    const int nParameters = K * 2;
    //for (int i = 0; i < nParameters; i++)
    //{
    //    parameters[i] = 1.0f / 3.0f;
    //}

    for (int iter = 0; iter < maxIterations; iter++)
    {
        sen::Mat<K * 3, K * 2> A;
        sen::Mat<K * 3, 1> b;

        float cost = 0.0f;

        for (int i = 0; i < nParameters; i++)
        {
            saka::dval parameters_optimizable[nParameters];
            for (int j = 0; j < nParameters; j++)
            {
                parameters_optimizable[j] = parameters[j];
            }
            parameters_optimizable[i].requires_grad();

            saka::dval3 vertices[K + 2];
            vertices[0] = saka::make_dval3(p_beg);
            vertices[K + 1] = saka::make_dval3(p_end);

            saka::dval3 shadingNormals[K];

            for (int k = 0; k < K; k++)
            {
                saka::dval param_u = parameters_optimizable[k * 2 + 0];
                saka::dval param_v = parameters_optimizable[k * 2 + 1];

                minimum_lbvh::Triangle tri = tris[k];

                float3 e0 = tri.vs[1] - tri.vs[0];
                float3 e1 = tri.vs[2] - tri.vs[0];
                vertices[k + 1] = saka::make_dval3(tri.vs[0]) + saka::make_dval3(e0) * param_u + saka::make_dval3(e1) * param_v;

                TriangleAttrib attrib = attribs[k];
                float3 ne0 = attrib.shadingNormals[1] - attrib.shadingNormals[0];
                float3 ne1 = attrib.shadingNormals[2] - attrib.shadingNormals[0];
                shadingNormals[k] = saka::make_dval3(attrib.shadingNormals[0]) + saka::make_dval3(ne0) * param_u + saka::make_dval3(ne1) * param_v;
            }

            for (int k = 0; k < K; k++)
            {
                saka::dval3 wi = vertices[k] - vertices[k + 1];
                saka::dval3 wo = vertices[k + 2] - vertices[k + 1];
                saka::dval3 n = shadingNormals[k];

                if (dot(wi, n).v < 0.0f)
                {
                    minimum_lbvh::swap(&wi, &wo);
                }


                saka::dval3 wo_optimize;
                if (eDescriptor.get(k) == Event::T)
                {
                    wo_optimize = saka::refraction_norm_free(wi, n, eta);
                }
                else
                {
                    wo_optimize = saka::reflection(wi, n);
                }

                saka::dval3 c = saka::cross(wo, wo_optimize);

                A(k * 3 + 0, i) = c.x.g;
                A(k * 3 + 1, i) = c.y.g;
                A(k * 3 + 2, i) = c.z.g;

                b(k * 3 + 0, 0) = c.x.v;
                b(k * 3 + 1, 0) = c.y.v;
                b(k * 3 + 2, 0) = c.z.v;

                if( i == 0 )
                {
                    cost += dot(c, c).v;
                }
            }
        }

        if (cost < costTolerance)
        {
            end_of_iter(iter, true);
            return true;
        }

        // SVD based solver
        // sen::Mat<K * 2, 1> dparams = sen::pinv(A) * b;

        // Householder QR based solver
        sen::Mat<K * 2, 1> dparams = sen::solve_qr_overdetermined(A, b);

        // Normal Equation and Cholesky Decomposition
        //sen::Mat<K * 2, K * 3> AT = sen::transpose(A);
        //sen::Mat<K * 2, K * 2> ATA = AT * A;
        //sen::Mat<K * 2, 1> dparams = sen::solve_cholesky(ATA, AT * b);

        for (int i = 0; i < nParameters; i++)
        {
            parameters[i] = parameters[i] - dparams(i, 0);
        }

        // extreme values can be rejected earlier
        if (4 /* it is an adhoc parameter but need some for thin triangles */ < iter)
        {
            for (int k = 0; k < K; k++)
            {
                float param_u = parameters[k * 2 + 0];
                float param_v = parameters[k * 2 + 1];

                // out side of triangle
                if (param_u < -0.5f || param_v < -0.5f || 1.5f < param_u + param_v)
                {
                    return false;
                }
            }
        }

        end_of_iter(iter, iter == maxIterations - 1);
    }
    return false;
}

PK_DEVICE inline saka::dval3 intersect_p_ray_plane(saka::dval3 ro, saka::dval3 rd, saka::dval3 ng, saka::dval3 v0)
{
    auto t = dot(v0 - ro, ng) / dot(ng, rd);
    return ro + rd * t;
};

template <int K, class callback = SolverEmptyCallback >
PK_DEVICE inline bool solveConstraints_v2(float parameters[K * 2], float3 p_beg, float3 p_end, minimum_lbvh::Triangle tris[K], TriangleAttrib attribs[K], float eta, EventDescriptor eDescriptor, int maxIterations, float costTolerance, callback end_of_iter = SolverEmptyCallback())
{
    //if (g_bruteforce)
    //{
    //    return solveConstraints<K>(parameters, p_beg, p_end, tris, attribs, eta, eDescriptor, maxIterations, costTolerance, end_of_iter);
    //}
    const int nParameters = K * 2;
    //for (int i = 0; i < nParameters; i++)
    //{
    //    parameters[i] = 1.0f / 3.0f;
    //}
    float d_params[2] = {};
    float3 firstCenter = tris[0].vs[0] +
       (tris[0].vs[1] - tris[0].vs[0]) * parameters[0] +
       (tris[0].vs[2] - tris[0].vs[0]) * parameters[1];
    float3 rd_base = normalize(firstCenter - p_beg);
    float3 T0 = tris[0].vs[1] - tris[0].vs[0];
    float3 T1 = tris[0].vs[2] - tris[0].vs[0];

    for (int iter = 0; iter < maxIterations; iter++)
    {
        sen::Mat<3, 2> A;
        sen::Mat<3, 1> b;

        float cost = 0.0f;

        for (int i = 0; i < 2; i++)
        {
            saka::dval differentials[2] = { d_params[0], d_params[1] };
            differentials[i].requires_grad();

            saka::dval3 ro = saka::make_dval3(p_beg);
            saka::dval3 rd = saka::make_dval3(rd_base)
                + saka::make_dval3(T0) * differentials[0]
                + saka::make_dval3(T1) * differentials[1];

            for (int j = 0; j < K; j++)
            {
                minimum_lbvh::Triangle tri = tris[j];
                TriangleAttrib attrib = attribs[j];

                float3 ng = minimum_lbvh::unnormalizedNormalOf(tri);

                saka::dval3 p = intersect_p_ray_plane(ro, rd, saka::make_dval3(ng), saka::make_dval3(tri.vs[0]));

                //pr::DrawLine(
                //    { ro.x.v, ro.y.v , ro.z.v },
                //    { p.x.v, p.y.v , p.z.v },
                //    { 255, 255, 0 },
                //    3
                //);

                saka::dval v = dot(p - saka::make_dval3(tri.vs[1]), saka::make_dval3(cross(ng, tri.vs[1] - tri.vs[0])));
                saka::dval w = dot(p - saka::make_dval3(tri.vs[2]), saka::make_dval3(cross(ng, tri.vs[2] - tri.vs[1])));
                saka::dval u = dot(p - saka::make_dval3(tri.vs[0]), saka::make_dval3(cross(ng, tri.vs[0] - tri.vs[2])));

                saka::dval area = u + v + w;
                u = u / area;
                v = v / area;
                w = w / area;

                parameters[j * 2] = u.v;
                parameters[j * 2 + 1] = v.v;

                saka::dval3 n =
                    saka::make_dval3(attrib.shadingNormals[0]) * w +
                    saka::make_dval3(attrib.shadingNormals[1]) * u +
                    saka::make_dval3(attrib.shadingNormals[2]) * v;

                saka::dval3 wi = -rd;

                saka::dval3 wo;
                if (eDescriptor.get(j) == Event::T)
                {
                    if (dot(wi, n).v < 0.0f)
                    {
                        wo = saka::refraction_norm_free(wi, -n, 1.0f / eta);
                        if (wo.x.v == 0.0f && wo.y.v == 0.0f && wo.z.v == 0.0f)
                        {
                            return false;
                        }
                    }
                    else
                    {
                        wo = saka::refraction_norm_free(wi, n, eta);
                    }
                }
                else
                {
                    wo = saka::reflection(wi, n);
                }

                ro = p;
                rd = wo;

                if (j + 1 == K)
                {
                    saka::dval3 c = saka::cross(wo, saka::make_dval3(p_end) - p /* the last vertex */);

                    A(0, i) = c.x.g;
                    A(1, i) = c.y.g;
                    A(2, i) = c.z.g;

                    b(0, 0) = c.x.v;
                    b(1, 0) = c.y.v;
                    b(2, 0) = c.z.v;

                    if (i == 0)
                    {
                        cost += dot(c, c).v;
                    }
                }
            }
        }

        if (cost < costTolerance)
        {
            end_of_iter(iter, true);
            return true;
        }

        // SVD based solver
        sen::SVD<3, 2> svd = sen::svd_BV(A);
        if (svd.converged == false)
        {
            return false;
        }
        sen::Mat<2, 1> dparams = svd.pinv() * b;

        // Householder QR based solver
        // sen::Mat<2, 1> dparams = sen::solve_qr_overdetermined(A, b);

        // Normal Equation and Cholesky Decomposition
        //sen::Mat<K * 2, K * 3> AT = sen::transpose(A);
        //sen::Mat<K * 2, K * 2> ATA = AT * A;
        //sen::Mat<K * 2, 1> dparams = sen::solve_cholesky(ATA, AT * b);

        //for (int i = 0; i < 2; i++)
        //{
        //    parameters[i] = parameters[i] - dparams(i, 0);
        //}
        for (int i = 0; i < 2; i++)
        {
            d_params[i] = d_params[i] - dparams(i, 0);
        }

        end_of_iter(iter, iter == maxIterations - 1);
    }
    return false;
}

PK_DEVICE inline float3 getVertex(int k, minimum_lbvh::Triangle tris[], float parameters[])
{
    minimum_lbvh::Triangle tri = tris[k];
    float3 e0 = tri.vs[1] - tri.vs[0];
    float3 e1 = tri.vs[2] - tri.vs[0];
    return tri.vs[0] + e0 * parameters[k * 2 + 0] + e1 * parameters[k * 2 + 1];
}

template <int K>
PK_DEVICE inline float contributableThroughput(float parameters[K * 2], float3 p_beg, float3 p_end, minimum_lbvh::Triangle tris[K], TriangleAttrib attribs[K], EventDescriptor eDescriptor, const minimum_lbvh::InternalNode* nodes, const minimum_lbvh::Triangle* sceneTriangles, minimum_lbvh::NodeIndex node, float eta )
{
    float3 vertices[K + 2];
    vertices[0] = p_beg;
    vertices[K + 1] = p_end;

    float3 shadingNormals[K + 2];

    for (int k = 0; k < K; k++)
    {
        float param_u = parameters[k * 2 + 0];
        float param_v = parameters[k * 2 + 1];

        // out side of triangle
        if (param_u < 0.0f || param_v < 0.0f || 1.0f < param_u + param_v)
        {
            return 0.0f;
        }

        vertices[k + 1] = getVertex(k, tris, parameters);

        TriangleAttrib attrib = attribs[k];
        float3 ne0 = attrib.shadingNormals[1] - attrib.shadingNormals[0];
        float3 ne1 = attrib.shadingNormals[2] - attrib.shadingNormals[0];
        shadingNormals[k + 1] = normalize(attrib.shadingNormals[0] + ne0 * param_u + ne1 * param_v);
    }

    shadingNormals[0] = normalize(vertices[1] - vertices[0]);
    shadingNormals[K + 1] = normalize(vertices[K] - vertices[K + 1]);

    float throughput = 1.0f;
    for (int k = 0; k < K; k++)
    {
        float3 wi = vertices[k] - vertices[k + 1];
        float3 wo = vertices[k + 2] - vertices[k + 1];
        float3 n = shadingNormals[k + 1];
        Event e = 0.0f < dot(wi, n) * dot(wo, n) ? Event::R : Event::T;
        if (eDescriptor.get(k) != e)
        {
            return 0.0f; // invalid event
        }
        float3 ng = minimum_lbvh::unnormalizedNormalOf(tris[k]);
        Event eg = 0.0f < dot(wi, ng) * dot(wo, ng) ? Event::R : Event::T;
        if (e != eg)
        {
            return 0.0f; // inconsistent
        }

        if (attribs[k].material == Material::Dielectric)
        {
            float reflectance = fresnel_exact_norm_free(wi, n, eta);
            throughput *= e == Event::R ? reflectance : 1.0f - reflectance;
        }
    }

    // check distance
    float eps = MIN_VERTEX_DIST;
    for (int i = 0; i < K + 1; i++)
    {
        float3 v0 = vertices[i];
        float3 v1 = vertices[i + 1];
        float3 d = v0 - v1;
        if (dot(d, d) < eps)
        {
            return 0.0f;
        }
    }

    // check occlusions
    for (int i = 0; i < K + 1; i++)
    {
        float3 v0 = vertices[i];
        float3 v1 = vertices[i + 1];
        float3 n0 = shadingNormals[i];
        float3 n1 = shadingNormals[i + 1];
        if (occluded(nodes, sceneTriangles, node, v0, n0, v1, n1))
        {
            return 0.0f;
        }
    }
    return throughput;
}

PK_DEVICE inline float dAdw(float3 ro, float3 rd, float3 p_end, minimum_lbvh::Triangle* tris, TriangleAttrib* attribs, EventDescriptor eDescriptor, int nEvent, float eta )
{
    rd = normalize(rd);

    auto intersect_p_ray_plane = [](saka::dval3 ro, saka::dval3 rd, saka::dval3 ng, saka::dval3 v0)
    {
        auto t = dot(v0 - ro, ng) / dot(ng, rd);
        return ro + rd * t;
    };

    float3 dAxis[2];
    for (int i = 0; i < 2; i++)
    {
        saka::dval differentials[2] = { 0.0f, 0.0f };
        differentials[i].requires_grad();

        saka::dval3 ro_j = saka::make_dval3(ro);

        float3 T0, T1;
        GetOrthonormalBasis(rd, &T0, &T1);

        saka::dval3 rd_j = saka::make_dval3(rd)
            + saka::make_dval3(T0) * differentials[0]
            + saka::make_dval3(T1) * differentials[1];

        bool inMedium = false;

        for (int j = 0; j < nEvent; j++)
        {
            minimum_lbvh::Triangle tri = tris[j];
            TriangleAttrib attrib = attribs[j];

            float3 ng = minimum_lbvh::unnormalizedNormalOf(tri);

            saka::dval3 p = intersect_p_ray_plane(ro_j, rd_j, saka::make_dval3(ng), saka::make_dval3(tri.vs[0]));

            saka::dval v = dot(p - saka::make_dval3(tri.vs[1]), saka::make_dval3(cross(ng, tri.vs[1] - tri.vs[0])));
            saka::dval w = dot(p - saka::make_dval3(tri.vs[2]), saka::make_dval3(cross(ng, tri.vs[2] - tri.vs[1])));
            saka::dval u = dot(p - saka::make_dval3(tri.vs[0]), saka::make_dval3(cross(ng, tri.vs[0] - tri.vs[2])));

            saka::dval area = u + v + w;
            u = u / area;
            v = v / area;
            w = w / area;

            saka::dval3 n =
                saka::make_dval3(attrib.shadingNormals[0]) * w +
                saka::make_dval3(attrib.shadingNormals[1]) * u +
                saka::make_dval3(attrib.shadingNormals[2]) * v;

            saka::dval3 wi = -rd_j;

            saka::dval3 wo;
            if (eDescriptor.get(j) == Event::T)
            {
                if (inMedium)
                {
                    wo = saka::refraction_norm_free(wi, -n, 1.0f / eta);
                    if (wo.x.v == 0.0f && wo.y.v == 0.0f && wo.z.v == 0.0f)
                    {
                        return 3.402823466e+38f;
                    }
                }
                else
                {
                    wo = saka::refraction_norm_free(wi, n, eta);
                }
                inMedium = !inMedium;
            }
            else
            {
                wo = saka::reflection(wi, n);
            }

            ro_j = p;
            rd_j = wo;
        }

        float3 p_last = { ro_j.x.v, ro_j.y.v, ro_j.z.v };
        saka::dval3 p_final = intersect_p_ray_plane(ro_j, rd_j, saka::make_dval3(p_last - p_end), saka::make_dval3(p_end));
        dAxis[i] = { p_final.x.g,  p_final.y.g,  p_final.z.g };

        //printf("x %.5f %.5f\n", p_final.x.v, p_end.x);
        //printf("y %.5f %.5f\n", p_final.y.v, p_end.y);
        //printf("z %.5f %.5f\n", p_final.z.v, p_end.z);
    }

    float len0 = length(dAxis[0]);
    float len1 = length(dAxis[1]);
    float sinTheta = length( cross(dAxis[0] / len0, dAxis[1] / len1) );
    float dAdwValue = len0 * len1 * fmaxf(sinTheta, 0.01f);

    //float3 crs = cross(dAxis[0], dAxis[1]);
    //float dAdwValue = sqrtf(fmaxf(dot(crs, crs), 1.0e-9f));
    return dAdwValue;
}

#define VISIBLE_SPECTRUM_MIN 390.0f
#define VISIBLE_SPECTRUM_MAX 830.0f

struct CauchyDispersion
{
    PK_DEVICE CauchyDispersion(float constantEta) : m_A(constantEta), m_B(0.0f)
    {
    }
    PK_DEVICE CauchyDispersion(float A, float B) : m_A(A), m_B(B)
    {
    }
    PK_DEVICE float operator()(float nm) const
    {
        return m_A + m_B / (nm * nm);
    }
    float m_A;
    float m_B;
};

PK_DEVICE inline CauchyDispersion BAF10_optical_glass()
{
    return CauchyDispersion(1.64732f, 7907.16861);
}
PK_DEVICE inline CauchyDispersion diamond()
{
    return CauchyDispersion(2.38272f, 12030.62f);
}

namespace CIE_2015_10deg
{
    PK_DEVICE inline float asymmetric_gaussian(float x, float mean, float sigma, float a) {
        float denom = sigma + a * (x - mean);
        if (denom < 1.0e-15f) { denom = 1.0e-15f; }
        if (sigma * 2.0f < denom) { denom = sigma * 2.0f; }
        float k = (x - mean) / denom;
        return expf(-k * k);
    }
    PK_DEVICE inline float cmf_x(float x) {
        float a = 0.42f * asymmetric_gaussian(x, 445.5849609375f, 31.146467208862305f, 0.06435633450746536f);
        float b = 1.16f * asymmetric_gaussian(x, 594.5570068359375f, 48.602108001708984f, -0.04772702232003212f);
        return (a + b - a * b * 42.559776306152344f) * 1.0008530432426956f;
    }
    PK_DEVICE inline float cmf_y(float x) {
        return 1.0f * asymmetric_gaussian(x, 556.8383178710938f, 66.54190826416016f, -0.026492968201637268f) * 1.004315086937574;
    }
    PK_DEVICE inline float cmf_z(float x) {
        return 2.146832f * asymmetric_gaussian(x, 445.9251708984375f, 30.91781997680664f, 0.08379141241312027f) * 0.9975112815948937;
    }

    PK_DEVICE inline float logistic_pdf(float x, float s)
    {
        float k = expf(-fabsf(x) / s);
        return k / ((1.0 + k) * (1.0 + k) * s);
    }
    PK_DEVICE inline float logistic_cdf(float x, float s)
    {
        return 1.0f / (1.0f + expf(-x / s));
    }
    PK_DEVICE inline float inverse_logistic_cdf(float u, float s)
    {
        if (0.99999994f < u) { u = 0.99999994f; }
        if (u < 1.175494351e-38f) { u = 1.175494351e-38f; }
        return -s * logf(1.0f / u - 1.0f);
    }
    PK_DEVICE inline float cmf_y_pdf(float x) {
        float sx = x - 554.270751953125f;
        float s = 26.879621505737305f;
        float a = -164.270751953125f;
        float b = 275.729248046875f;
        return logistic_pdf(sx, s) / (logistic_cdf(b, s) - logistic_cdf(a, s));
    }

    PK_DEVICE inline float cmf_y_sample(float u) {
        float s = 26.879621505737305f;
        float a = -164.270751953125f;
        float b = 275.729248046875f;
        float Pa = logistic_cdf(a, s);
        float Pb = logistic_cdf(b, s);
        return inverse_logistic_cdf(Pa + (Pb - Pa) * u, s) + 554.270751953125f;
    }
}

PK_DEVICE inline float3 xyz2srgblinear(float3 xyz)
{
    float3 srgblinear = {
		dot({3.2406f, -1.5372f, -0.4986}, xyz),
		dot({-0.9689f,  1.8758f,  0.0415f}, xyz),
		dot({0.0557f, -0.204f,   1.057f}, xyz),
	};
    return srgblinear;
}
PK_DEVICE inline float srgb_oetf( float r )
{
    r = fmaxf(r, 0.0f);
    if (r <= 0.003130f)
    {
        return 12.92f * r;
    }
    return 1.055f * powf(r, 1.0f / 2.4f) - 0.055f;
}

#define INTEGRAL_OF_CMF_Y_IN_NM 118.51810464018001f


class Texture8RGBX
{
public:
#if !defined(PK_KERNELCC)
    Texture8RGBX():m_buffer(TYPED_BUFFER_DEVICE), m_width(0), m_height(0){}
#endif
    PK_DEVICE float3 samplePoint(float u, float v) const
    {
        u = u - floor(u); // repeat
        v = v - floor(v); // repeat

        int ix = fminf( u * m_width, m_width - 1.0f);
        int iy = fminf( v * m_height, m_height - 1.0f);
        int index = ix + m_width * iy;
        uint8_t r = m_buffer[index * 4];
        uint8_t g = m_buffer[index * 4 + 1];
        uint8_t b = m_buffer[index * 4 + 2];
        float3 color = { powf( r / 255.0f, 2.2f), powf(g / 255.0f, 2.2f), powf(b / 255.0f, 2.2f) };
        return color;
    }
    TypedBuffer<uint8_t> m_buffer;
    uint32_t m_width;
    uint32_t m_height;
};


#define STACK_BUFFER_MAX_WARPS ( 256 * 256 )
#define STACK_BUFFER_MAX_ELEMENT 128

#if !defined(PK_KERNELCC)
class StackBufferAllocator
{
public:
    StackBufferAllocator()
    {
        m_stackBuffer.allocate(STACK_BUFFER_MAX_WARPS * 32 * STACK_BUFFER_MAX_ELEMENT);
        oroMemsetD32(m_stackBuffer.data(), 0, m_stackBuffer.size());
    }
    minimum_lbvh::NodeIndex* data() { return m_stackBuffer.data(); }
private:
    TypedBuffer<minimum_lbvh::NodeIndex> m_stackBuffer = TypedBuffer<minimum_lbvh::NodeIndex>(TYPED_BUFFER_DEVICE);
};
#else
class StackBufferWarp
{
public:
    PK_DEVICE StackBufferWarp(minimum_lbvh::NodeIndex *stackBuffer):m_stackBuffer(stackBuffer)
    {
        int linearBlockDim = blockDim.x * blockDim.y * blockDim.z;
        int linearThreadIdx = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
        int linearBlockIdx = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
        int warpsInBlock = linearBlockDim / 32;
        int laneIdx = linearThreadIdx % 32;
        int warpIdx = linearThreadIdx / 32;
        int globalWarpIdx = linearBlockIdx * warpsInBlock + warpIdx;

        uint32_t index = (globalWarpIdx % STACK_BUFFER_MAX_WARPS) * 32 * STACK_BUFFER_MAX_ELEMENT;
        bool success = false;
        while(success == false)
        {
            if (laneIdx == 0)
            {
                success = atomicCAS((uint32_t*)&m_stackBuffer[index], 0 /*compare*/, 32) == 0;
            }
            success = __any(success);
        }
        m_baseIndex = index;
    }
    PK_DEVICE ~StackBufferWarp()
    {
        atomicDec((uint32_t*)&m_stackBuffer[m_baseIndex], 0xFFFFFFFF);
    }
    PK_DEVICE minimum_lbvh::NodeIndex* stack()
    {
        int linearThreadIdx = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
        int laneIdx = linearThreadIdx % 32;
        return &m_stackBuffer[m_baseIndex + laneIdx * STACK_BUFFER_MAX_ELEMENT + 1];
    }
    minimum_lbvh::NodeIndex* m_stackBuffer;
    uint32_t m_baseIndex;
};
#endif


#if !defined(PK_KERNELCC)

inline glm::vec3 to(float3 p)
{
    return { p.x, p.y, p.z };
}
inline float3 to(glm::vec3 p)
{
    return { p.x, p.y, p.z };
}

inline void loadTexture(Texture8RGBX* tex, const char* file)
{
    pr::Image2DRGBA8 image;
    image.load(file);
    tex->m_buffer.allocate(image.bytes());
    tex->m_width = image.width();
    tex->m_height = image.height();
    oroMemcpyHtoD(tex->m_buffer.data(), image.data(), image.bytes());
}

class DeviceStopwatch
{
public:
    DeviceStopwatch(oroStream stream)
    {
        m_stream = stream;
        oroEventCreateWithFlags(&m_start, oroEventDefault);
        oroEventCreateWithFlags(&m_stop, oroEventDefault);
    }
    ~DeviceStopwatch()
    {
        oroEventDestroy(m_start);
        oroEventDestroy(m_stop);
    }
    DeviceStopwatch(const DeviceStopwatch&) = delete;
    void operator=(const DeviceStopwatch&) = delete;

    void start() { oroEventRecord(m_start, m_stream); }
    void stop() { oroEventRecord(m_stop, m_stream); }

    float getElapsedMs() const
    {
        oroEventSynchronize(m_stop);
        float ms = 0;
        oroEventElapsedTime(&ms, m_start, m_stop);
        return ms;
    }
private:
    oroStream m_stream;
    oroEvent m_start;
    oroEvent m_stop;
};

class PKRenderer
{
public:
    PKRenderer()
    {
    }
    void setup(oroDevice device, const char* scene, const char* kernelDir)
    {
        using namespace pr;

        std::vector<std::string> options;
        options.push_back("-I");
        options.push_back(kernelDir);

        m_shader = std::unique_ptr<Shader>(new Shader(JoinPath(kernelDir, "main_gpu.cu").c_str(), "main_gpu", options));
        m_onesweep = std::unique_ptr<tinyhiponesweep::OnesweepSort>(new tinyhiponesweep::OnesweepSort(device));
        m_gpuBuilder = std::unique_ptr< minimum_lbvh::BVHGPUBuilder>(new minimum_lbvh::BVHGPUBuilder(JoinPath(kernelDir, "minimum_lbvh.cu").c_str(), kernelDir));

        std::string err;
        m_archive.open(scene, err);
    
        loadTexture(&m_floorTex, GetDataPath("assets/laminate_floor_02_diff_2k.jpg").c_str());
    }

    void loadFrame(int frame)
    {
        using namespace pr;

        std::vector<minimum_lbvh::Triangle> triangles;
        std::vector<TriangleAttrib> triangleAttribs;

        std::string err;
        std::shared_ptr<FScene> scene = m_archive.readFlat(frame, err);

        scene->visitCamera([&](std::shared_ptr<const pr::FCameraEntity> cameraEntity) {
            if (!cameraEntity->visible())
            {
                return;
            }
            if (m_loadCamera == false)
            {
                return;
            }
            m_camera = cameraFromEntity(cameraEntity.get());
        });

        scene->visitPolyMesh([&](std::shared_ptr<const FPolyMeshEntity> polymesh) {
            if (polymesh->visible() == false)
            {
                return;
            }
            std::string name = polymesh->fullname();
            if (name == "/light/pointlight")
            {
                m_p_light = polymesh->localToWorld() * glm::vec4(0, 0, 0, 1);
                return;
            }

            AttributeSpreadsheet* details = polymesh->attributeSpreadsheet(AttributeSpreadsheetType::Details);
            Material material = Material::Diffuse;
            if (auto matCol = details->columnAsString("material"))
            {
                const std::string& matString = details->columnAsString("material")->get(0);
                if (matString == "mirror")
                {
                    material = Material::Mirror;
                }
                else if (matString == "dielectric")
                {
                    material = Material::Dielectric;
                }
            }

            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());
            ColumnView<glm::vec3> normals(polymesh->normals());

            int indexBase = 0;
            for (int i = 0; i < faceCounts.count(); i++)
            {
                int nVerts = faceCounts[i];
                PR_ASSERT(nVerts == 3);
                minimum_lbvh::Triangle tri;
                TriangleAttrib attrib;
                attrib.material = material;
                for (int j = 0; j < nVerts; ++j)
                {
                    glm::vec3 p = positions[indices[indexBase + j]];
                    tri.vs[j] = { p.x, p.y, p.z };

                    glm::vec3 ns = normals[indexBase + j];
                    attrib.shadingNormals[j] = { ns.x, ns.y, ns.z };
                }

                float3 e0 = tri.vs[1] - tri.vs[0];
                float3 e1 = tri.vs[2] - tri.vs[1];
                float3 e2 = tri.vs[0] - tri.vs[2];

                triangles.push_back(tri);
                triangleAttribs.push_back(attrib);
                indexBase += nVerts;
            }
        });

        m_trianglesDevice << triangles;
        m_triangleAttribsDevice << triangleAttribs;

        m_gpuBuilder->build(m_trianglesDevice.data(), m_trianglesDevice.size(), 0, *m_onesweep, 0 /*stream*/);
    }
    int frameCount() const
    {
        return m_archive.frameCount();
    }

    void allocate(int imageWidth, int imageHeight)
    {
        m_imageWidth = imageWidth;
        m_imageHeight = imageHeight;


        if (m_pixels.size() != imageWidth * imageHeight)
        {
            m_pathCache.init(0.01f);

            m_debugPoints.allocate(1 << 22);
            m_debugPointCount.allocate(1);

            m_pixels.allocate(imageWidth * imageHeight);
            m_accumulators.allocate(imageWidth * imageHeight);
            m_firstDiffuses.allocate(imageWidth * imageHeight);
        }
    }

    void step()
    {
        using namespace pr;

        //printf("---\n");
        //DeviceStopwatch sw(0);
        //sw.start();

        CauchyDispersion cauchy = BAF10_optical_glass();

        RayGenerator rayGenerator;
        rayGenerator.lookat(to(m_camera.origin), to(m_camera.lookat), to(m_camera.up), m_camera.fovy, m_imageWidth, m_imageHeight);

        m_shader->launch("solvePrimary",
            ShaderArgument()
            .value(m_accumulators.data())
            .value(m_firstDiffuses.data())
            .value(int2{ m_imageWidth, m_imageHeight })
            .value(rayGenerator)
            .value(m_gpuBuilder->m_rootNode)
            .value(m_gpuBuilder->m_internals)
            .value(m_trianglesDevice.data())
            .value(m_triangleAttribsDevice.data())
            .value(to(m_p_light))
            .value(m_lightIntencity)
            .value(m_radianceClamp)
            .value(cauchy)
            .ptr(&m_floorTex)
            .value(m_iteration)
            .value(m_stackBufferAllocator.data()),
            div_round_up64(m_imageWidth, 16), div_round_up64(m_imageHeight, 16), 1,
            16, 16, 1,
            0
        );

        //sw.stop();
        //printf("solvePrimary %f\n", sw.getElapsedMs());

        auto solveSpecular = [&](int K, EventDescriptor eDescriptor) {
            oroMemsetD32(m_debugPointCount.data(), 0, 1);
            m_pathCache.clear();

            //sw.start();

            char photonTrace[128];
            char solveSpecular[128];
            sprintf(photonTrace, "photonTrace_K%d", K);
            sprintf(solveSpecular, "solveSpecular_K%d", K);

            m_shader->launch(photonTrace,
                ShaderArgument()
                .value(m_gpuBuilder->m_rootNode)
                .value(m_gpuBuilder->m_internals)
                .value(m_trianglesDevice.data())
                .value(m_triangleAttribsDevice.data())
                .value(to(m_p_light))
                .value(eDescriptor)
                .value(cauchy(VISIBLE_SPECTRUM_MIN))
                .value(cauchy(VISIBLE_SPECTRUM_MAX))
                .value(m_iteration)
                .ptr(&m_pathCache)
                .value(m_minThroughput)
                .value(m_debugPoints.data())
                .value(m_debugPointCount.data()),
                m_gpuBuilder->m_nTriangles, 1, 1,
                32, 1, 1,
                0
            );

            //sw.stop();
            //printf("%s %f\n", photonTrace, sw.getElapsedMs());

            //printf(" occ %f\n", m_pathCache.occupancy());

            // debug view
            if (0)
            {
                int nPoints = 0;
                oroMemcpyDtoH(&nPoints, m_debugPointCount.data(), sizeof(int));
                std::vector<float3> points(nPoints);
                oroMemcpyDtoH(points.data(), m_debugPoints.data(), sizeof(float3) * nPoints);
                for (int i = 0; i < nPoints; i++)
                {
                    DrawPoint(to(points[i]), { 255, 0, 0 }, 2);
                }
            }

            //sw.start();

            m_shader->launch(solveSpecular,
                ShaderArgument()
                .value(m_accumulators.data())
                .value(m_firstDiffuses.data())
                .value(int2{ m_imageWidth, m_imageHeight })
                .value(m_gpuBuilder->m_rootNode)
                .value(m_gpuBuilder->m_internals)
                .value(m_trianglesDevice.data())
                .value(m_triangleAttribsDevice.data())
                .value(to(m_p_light))
                .value(m_lightIntencity)
                .ptr(&m_pathCache)
                .value(eDescriptor)
                .value(cauchy)
                .value(m_iteration),
                div_round_up64(m_imageWidth, 16), div_round_up64(m_imageHeight, 16), 1,
                16, 16, 1,
                0
            );

            //sw.stop();
            //printf("%s %f\n", solveSpecular, sw.getElapsedMs());
        };

        solveSpecular(1, { Event::T });
        solveSpecular(1, { Event::R });
        solveSpecular(2, { Event::T, Event::T });
        solveSpecular(4, { Event::T, Event::T, Event::T, Event::T });

        m_iteration++;
    }

    void clear()
    {
        oroMemsetD8(m_accumulators.data(), 0, m_accumulators.size() * sizeof(float4));
        m_iteration = 0;
    }

    void resolve(pr::Image2DRGBA8 *imageOut)
    {
        m_shader->launch("pack",
            ShaderArgument()
            .value(m_pixels.data())
            .value(m_accumulators.data())
            .value(m_imageWidth * m_imageHeight),
            div_round_up64(m_imageWidth * m_imageHeight, 256), 1, 1,
            256, 1, 1,
            0
        );

        imageOut->allocate(m_imageWidth, m_imageHeight);
        oroMemcpyDtoH(imageOut->data(), m_pixels.data(), m_pixels.bytes());
    }

    std::unique_ptr<Shader> m_shader;
    std::unique_ptr<tinyhiponesweep::OnesweepSort> m_onesweep;
    std::unique_ptr< minimum_lbvh::BVHGPUBuilder> m_gpuBuilder;

    TypedBuffer<uint32_t> m_pixels = TypedBuffer<uint32_t>(TYPED_BUFFER_DEVICE);
    TypedBuffer<float4> m_accumulators = TypedBuffer<float4>(TYPED_BUFFER_DEVICE);

    TypedBuffer<minimum_lbvh::Triangle> m_trianglesDevice = TypedBuffer<minimum_lbvh::Triangle>(TYPED_BUFFER_DEVICE);
    TypedBuffer<TriangleAttrib> m_triangleAttribsDevice = TypedBuffer<TriangleAttrib>(TYPED_BUFFER_DEVICE);

    PathCache m_pathCache = PathCache(TYPED_BUFFER_DEVICE);
    TypedBuffer<FirstDiffuse> m_firstDiffuses = TypedBuffer<FirstDiffuse>(TYPED_BUFFER_DEVICE);

    Texture8RGBX m_floorTex;

    TypedBuffer<float3> m_debugPoints = TypedBuffer<float3>(TYPED_BUFFER_DEVICE);
    TypedBuffer<int> m_debugPointCount = TypedBuffer<int>(TYPED_BUFFER_DEVICE);
    StackBufferAllocator m_stackBufferAllocator;

    int m_imageWidth = 0;
    int m_imageHeight = 0;
    pr::AbcArchive m_archive;
    glm::vec3 m_p_light = { -0.804876, 0.121239, -1.58616 };
    float m_lightIntencity = 5.0f;
    float m_minThroughput = 0.05f;
    float m_radianceClamp = 5.0f;
    pr::Camera3D m_camera;

    bool m_loadCamera = true;
    int m_iteration = 0;
};

struct InternalNormalBound
{
    minimum_lbvh::AABB normalBounds[2];
    uint32_t counter;
};

#if defined(ENABLE_PATH_CUTS)

inline interval::intr3 toIntr3(minimum_lbvh::AABB aabb)
{
    return {
        {aabb.lower.x, aabb.upper.x},
        {aabb.lower.y, aabb.upper.y},
        {aabb.lower.z, aabb.upper.z}
    };
}

template <int K>
struct AdmissibleTriangles
{
    int indices[K];
};

template <int K>
struct AdmissibleNodes
{
    minimum_lbvh::NodeIndex nodes[K];

    static AdmissibleNodes<K> invalid()
    {
        AdmissibleNodes<K> admissible;
        for (int i = 0; i < K; i++)
        {
            admissible.nodes[i] = minimum_lbvh::NodeIndex::invalid();
        }
        return admissible;
    }
};

inline minimum_lbvh::AABB nodeAABB(const  minimum_lbvh::InternalNode& node)
{
    minimum_lbvh::AABB bounds = node.aabbs[0];
    bounds.extend(node.aabbs[1]);
    return bounds;
}

inline void dump(interval::intr3 v)
{
    printf("{%.8f, %.8f},\n", v.x.l, v.x.u);
    printf("{%.8f, %.8f},\n", v.y.l, v.y.u);
    printf("{%.8f, %.8f},\n", v.z.l, v.z.u);
}

extern bool g_bruteforce;

template <int K, class callback>
inline void traverseAdmissibleNodes(EventDescriptor admissibleEvents, float eta, float3 p_beg, float3 p_end, minimum_lbvh::InternalNode *internals, InternalNormalBound* internalsNormalBound, minimum_lbvh::Triangle* tris, TriangleAttrib* attribs, minimum_lbvh::NodeIndex node, callback admissibles )
{
    interval::intr3 p_beg_intr = interval::make_intr3(p_beg);
    interval::intr3 p_end_intr = interval::make_intr3(p_end);

    //std::stack<minimum_lbvh::NodeIndex> stack;
    //stack.push(minimum_lbvh::NodeIndex::invalid());
    //minimum_lbvh::NodeIndex currentNode = node;

    //while (currentNode != minimum_lbvh::NodeIndex::invalid())
    //{
    //    if (currentNode.m_isLeaf)
    //    {
    //        AdmissibleTriangles<K> admissibleTriangles;
    //        admissibleTriangles.indices[0] = currentNode.m_index;
    //        admissibles(admissibleTriangles);
    //    }
    //    else
    //    {
    //        for (int i = 0; i < 2; i++)
    //        {
    //            interval::intr3 triangle_intr = toIntr3(internals[currentNode.m_index].aabbs[i]);
    //            interval::intr3 wi_intr = p_to - triangle_intr;
    //            interval::intr3 wo_intr = p_from - triangle_intr;
    //            interval::intr3 normal_intr = toIntr3(internalsNormalBound[currentNode.m_index].normalBounds[i]);
    //            interval::intr3 R = interval::reflection(wi_intr, normal_intr);
    //            interval::intr3 c = interval::cross(R, wo_intr);

    //            if (interval::zeroIncluded(c))
    //            {
    //                stack.push(internals[currentNode.m_index].children[i]);
    //            }
    //        }
    //    }

    //    currentNode = stack.top(); stack.pop();
    //}

    std::stack<AdmissibleNodes<K>> stack;
    stack.push(AdmissibleNodes<K>::invalid());

    AdmissibleNodes<K> currentNode;
    for (int i = 0; i < K; i++)
    {
        currentNode.nodes[i] = node;
    }

    while (currentNode.nodes[0] != minimum_lbvh::NodeIndex::invalid())
    {
        int cutIndex = -1;
        float maxSA = -1.0f;
        for (int i = 0; i < K; i++)
        {
            minimum_lbvh::NodeIndex node = currentNode.nodes[i];
            if (node.m_isLeaf)
            {
                continue;
            }
            
            minimum_lbvh::AABB bounds = nodeAABB(internals[node.m_index]);
            float sa = bounds.surfaceArea();
            if (maxSA < sa)
            {
                cutIndex = i;
                maxSA = sa;
            }
        }

        if (cutIndex == -1) // meaning all of them are leaf.
        {
            AdmissibleTriangles<K> admissibleTriangles;
            for (int i = 0; i < K; i++)
            {
                admissibleTriangles.indices[i] = currentNode.nodes[i].m_index;
            }

            bool invalid = false;
            for (int i = 0; i < K - 1; i++)
            {
                if (admissibleTriangles.indices[i] == admissibleTriangles.indices[i + 1])
                {
                    invalid = true;
                    break;
                }
            }

            // Geometric constraints at beg and end when mesh is closed
            {
                int index_beg = admissibleTriangles.indices[0];
                minimum_lbvh::Triangle tri_beg = tris[index_beg];
                float3 ng = minimum_lbvh::unnormalizedNormalOf(tri_beg);

                // light weight winding correction
                ng *= dot(ng, attribs[index_beg].shadingNormals[0]);

                if (dot(ng, p_beg) < dot(ng, tri_beg.vs[0]))
                {
                    invalid = true;
                }
            }
            {
                int index_end = admissibleTriangles.indices[K - 1];
                minimum_lbvh::Triangle tri_end = tris[index_end];
                float3 ng = minimum_lbvh::unnormalizedNormalOf(tri_end);

                // light weight winding correction
                ng *= dot(ng, attribs[index_end].shadingNormals[0]);

                if (dot(ng, p_end) < dot(ng, tri_end.vs[0]))
                {
                    invalid = true;
                }
            }

            if (invalid == false)
            {
                ////bool debug = admissibleTriangles.indices[0] == 13 && admissibleTriangles.indices[1] == 91;
                //bool debug = admissibleTriangles.indices[0] == 25 && admissibleTriangles.indices[1] == 91;
                //if (debug)
                //{
                //    printf("");
                //}
                //
                //interval::intr3 vertices[K + 2];
                //interval::intr3 normals[K];
                //vertices[0] = p_beg_intr;
                //vertices[K + 1] = p_end_intr;

                //for (int k = 0; k < K; k++)
                //{
                //    int index = admissibleTriangles.indices[k];
                //    
                //    minimum_lbvh::AABB bound;
                //    bound.setEmpty();
                //    for (float3 p : tris[index].vs)
                //    {
                //        bound.extend(p);
                //    }
                //    vertices[k + 1] = toIntr3(bound);

                //    minimum_lbvh::AABB nbound;
                //    nbound.setEmpty();
                //    for (float3 p : attribs[index].shadingNormals)
                //    {
                //        nbound.extend(p);
                //    }
                //    normals[k] = toIntr3(nbound);
                //}

                //bool admissible = true;
                //bool inMedium = false;
                //interval::intr3 wi_intr = vertices[0] - vertices[1];
                //for (int k = 0; k < K; k++)
                //{
                //    interval::intr3 wo_intr = vertices[k + 2] - vertices[k + 1];
                //    interval::intr3 normal_intr = normals[k];

                //    interval::intr3 wo_next;
                //    if (inMedium)
                //    {
                //        if( interval::refraction_norm_free(&wo_next, wi_intr, -normal_intr, 1.0f / eta) == false )
                //        {
                //            admissible = false;
                //            break;
                //        }
                //    }
                //    else
                //    {
                //        if (interval::refraction_norm_free(&wo_next, wi_intr, normal_intr, eta) == false)
                //        {
                //            admissible = false;
                //            break;
                //        }
                //    }

                //    if (interval::zeroIncluded(interval::cross(wo_next, wo_intr)) == false)
                //    {
                //        admissible = false;
                //        break;
                //    }

                //    inMedium = !inMedium;
                //    wi_intr = -wo_next;

                //    if (debug)
                //    {
                //        auto DrawAABB = [](interval::intr3 bound, glm::u8vec3 c, float lineWidth)
                //        {
                //            pr::DrawAABB({ bound.x.l, bound.y.l, bound.z.l }, { bound.x.u, bound.y.u, bound.z.u }, c, lineWidth);
                //        };

                //        //if (k == 0)
                //        //{

                //        //    DrawAABB(vertices[k + 1], { 255, 0, 0 }, 2);
                //        //    DrawAABB(vertices[k + 2], { 0, 255, 0 }, 2);
                //        //}

                //        //if (k == 0)
                //        //{
                //        //    dump(normal_intr);
                //        //    dump(wi_intr);
                //        //    dump(wo_intr);
                //        //}
                //    }
                //}

                ////std::reverse(vertices, vertices + 4);
                ////std::reverse(normals, normals + 2);

                ////wi_intr = vertices[0] - vertices[1];
                ////for (int k = 0; k < K; k++)
                ////{
                ////    interval::intr3 wo_intr = vertices[k + 2] - vertices[k + 1];
                ////    interval::intr3 normal_intr = normals[k];

                ////    interval::intr3 wo_next;
                ////    if (inMedium)
                ////    {
                ////        if (interval::refraction_norm_free(&wo_next, wi_intr, -normal_intr, 1.0f / eta) == false)
                ////        {
                ////            admissible = false;
                ////            break;
                ////        }
                ////    }
                ////    else
                ////    {
                ////        if (interval::refraction_norm_free(&wo_next, wi_intr, normal_intr, eta) == false)
                ////        {
                ////            admissible = false;
                ////            break;
                ////        }
                ////    }

                ////    if (interval::zeroIncluded(interval::cross(wo_next, wo_intr)) == false)
                ////    {
                ////        admissible = false;
                ////        break;
                ////    }

                ////    inMedium = !inMedium;
                ////    wi_intr = -wo_next;
                ////}

                ////if (admissible)
                {
                    admissibles(admissibleTriangles);
                }
            }
        }
        else
        {
            for (int i = 0; i < 2; i++)
            {
#if 0
                interval::intr3 triangle_intr = toIntr3(internals[currentNode.nodes[cutIndex].m_index].aabbs[i]);
                interval::intr3 prev_vert, next_vert;
                if (cutIndex == 0)
                {
                    prev_vert = p_beg_intr;
                }
                else
                {
                    minimum_lbvh::NodeIndex prev_node = currentNode.nodes[cutIndex - 1];
                    if (prev_node.m_isLeaf)
                    {
                        minimum_lbvh::AABB bound;
                        bound.setEmpty();
                        for (float3 p : tris[prev_node.m_index].vs)
                        {
                            bound.extend(p);
                        }
                        prev_vert = toIntr3(bound);
                    }
                    else
                    {
                        prev_vert = toIntr3(nodeAABB(internals[prev_node.m_index]));
                    }
                }

                if (cutIndex == K - 1)
                {
                    next_vert = p_end_intr;
                }
                else
                {
                    minimum_lbvh::NodeIndex next_node = currentNode.nodes[cutIndex + 1];
                    if (next_node.m_isLeaf)
                    {
                        minimum_lbvh::AABB bound;
                        bound.setEmpty();
                        for (float3 p : tris[next_node.m_index].vs)
                        {
                            bound.extend(p);
                        }
                        next_vert = toIntr3(bound);
                    }
                    else
                    {
                        next_vert = toIntr3(nodeAABB(internals[next_node.m_index]));
                    }
                }

                interval::intr3 wi_intr = next_vert - triangle_intr;
                interval::intr3 wo_intr = prev_vert - triangle_intr;
                interval::intr3 normal_intr = toIntr3(internalsNormalBound[currentNode.nodes[cutIndex].m_index].normalBounds[i]); /* this never be leaf */
                
                bool admissible = false;
                if (admissibleEvents.get(k) == Event::R) // maybe process always later
                {
                    interval::intr3 R = interval::reflection(wi_intr, normal_intr);
                    interval::intr3 Rc = interval::cross(R, wo_intr);

                    if (interval::zeroIncluded(Rc))
                    {
                        admissible = true;
                    }
                }
                else
                {
                    interval::intr s = dot(wi_intr, normal_intr) * dot(wo_intr, normal_intr);
                    if (s.l < 0.0f) // could be T event
                    {
                        interval::intr3 wi = interval::normalize(wi_intr);
                        interval::intr3 wo = interval::normalize(wo_intr);
                        interval::intr3 ht0 = wi * eta + wo;
                        interval::intr3 ht1 = wo * eta + wi;
                        if (interval::zeroIncluded(interval::cross(ht0, normal_intr)) || interval::zeroIncluded(interval::cross(ht1, normal_intr)))
                        {
                            admissible = true;
                        }
                    }

                    if (g_bruteforce)
                    {
                        admissible = true;
                    }
                }
#else

                interval::intr3 vertices[K + 2];
                interval::intr3 normals[K];
                vertices[0] = p_beg_intr;
                vertices[K + 1] = p_end_intr;

                for (int k = 0; k < K; k++)
                {
                    minimum_lbvh::NodeIndex node = currentNode.nodes[k];

                    if (k == cutIndex)
                    {
                        // node is always internal node
                        vertices[k + 1] = toIntr3(internals[node.m_index].aabbs[i]); 
                        normals[k] = toIntr3(internalsNormalBound[node.m_index].normalBounds[i]);
                    }
                    else if(node.m_isLeaf)
                    {
                        minimum_lbvh::AABB bound;
                        bound.setEmpty();
                        for (float3 p : tris[node.m_index].vs)
                        {
                            bound.extend(p);
                        }
                        vertices[k + 1] = toIntr3(bound);

                        minimum_lbvh::AABB nbound;
                        nbound.setEmpty();
                        for (float3 p : attribs[node.m_index].shadingNormals)
                        {
                            nbound.extend(p);
                        }
                        normals[k] = toIntr3(nbound);
                    }
                    else
                    {
                        vertices[k + 1] = toIntr3(nodeAABB(internals[node.m_index]));

                        minimum_lbvh::AABB nbound;
                        nbound.setEmpty();
                        nbound.extend(internalsNormalBound[node.m_index].normalBounds[0]);
                        nbound.extend(internalsNormalBound[node.m_index].normalBounds[1]);
                        normals[k] = toIntr3(nbound);
                    }
                }

                bool admissible = true;
                bool inMedium = false;
                interval::intr3 wi_intr = vertices[0] - vertices[1];
                for (int k = 0; k < K; k++)
                {
                    interval::intr3 wo_intr = vertices[k + 2] - vertices[k + 1];
                    interval::intr3 normal_intr = normals[k];

                    interval::intr3 wo_next;

                    if (admissibleEvents.get(k) == Event::R) // maybe process always later
                    {
                        wo_next = interval::reflection(wi_intr, normal_intr);
                    }
                    else
                    {
                        if (inMedium)
                        {
                            interval::intr3 wi = vertices[k] - vertices[k + 1];
                            interval::intr3 wo = wo_intr;
                            interval::intr3 ht = refraction_normal(wi, wo, 1.0f / eta);
                            if (interval::zeroIncluded(cross(normal_intr, ht)) == false)
                            {
                                admissible = false;
                                break;
                            }

                            if (interval::refraction_norm_free(&wo_next, wi_intr, -normal_intr, 1.0f / eta) == false)
                            {
                                admissible = false;
                                break;
                            }
                        }
                        else
                        {
                            interval::intr3 wi = vertices[k] - vertices[k + 1];
                            interval::intr3 wo = wo_intr;
                            interval::intr3 ht = refraction_normal(wi, wo, eta);
                            if (interval::zeroIncluded(cross(normal_intr, ht)) == false)
                            {
                                admissible = false;
                                break;
                            }

                            if (interval::refraction_norm_free(&wo_next, wi_intr, normal_intr, eta) == false)
                            {
                                admissible = false;
                                break;
                            }
                        }
                    }

                    if (interval::zeroIncluded(interval::cross(wo_next, wo_intr)) == false)
                    {
                        admissible = false;
                        break;
                    }

                    inMedium = !inMedium;
                    wi_intr = -wo_next;
                }
#endif

                if (admissible)
                {
                    AdmissibleNodes<K> newNodes = currentNode;
                    newNodes.nodes[cutIndex] = internals[currentNode.nodes[cutIndex].m_index].children[i];
                    stack.push(newNodes);
                }
            }
        }

        currentNode = stack.top(); stack.pop();
    }
}

#endif

#endif