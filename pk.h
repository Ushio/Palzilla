#include <math.h>
#include "helper_math.h"
#include "saka.h"
#include "sen.h"
#include "minimum_lbvh.h"

enum class Material
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

inline float3 reflection(float3 wi, float3 n)
{
    return n * dot(wi, n) * 2.0f / dot(n, n) - wi;
}

inline float fresnel_exact(float3 wi, float3 n, float eta /* eta_t / eta_i */) {
    float c = dot(n, wi);
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

inline float fresnel_exact_norm_free(float3 wi, float3 n, float eta /* eta_t / eta_i */) {
    float nn = dot(n, n);
    float wiwi = dot(wi, wi);
    float C = dot(n, wi);
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

inline float3 refraction_norm_free(float3 wi, float3 n, float eta /* = eta_t / eta_i */)
{
    float NoN = dot(n, n);
    float WIoN = dot(wi, n);
    float WIoWI = dot(wi, wi);
    float k = NoN * WIoWI * (eta * eta - 1.0f) + WIoN * WIoN;
    return -wi * NoN + (WIoN - sqrtf(k)) * n;
}

// Building an Orthonormal Basis, Revisited
void GetOrthonormalBasis(float3 zaxis, float3* xaxis, float3* yaxis) {
    const float sign = copysignf(1.0f, zaxis.z);
    const float a = -1.0f / (sign + zaxis.z);
    const float b = zaxis.x * zaxis.y * a;
    *xaxis = { 1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x };
    *yaxis = { b, sign + zaxis.y * zaxis.y * a, -zaxis.y };
}

struct SolverEmptyCallback {
    void operator()( int iter, bool converged ) const{}
};

enum class Event
{
    R = 0,
    T = 1
};
struct EventDescriptor
{
    EventDescriptor() : m_events(0) {}

    Event get(uint32_t index) const {
        bool setbit = m_events & (1u << index);
        return setbit ? Event::T : Event::R;
    }
    void set(uint32_t index, Event e)
    {
        m_events &= ~(1u << index);
        if (e == Event::T)
        {
            m_events |= 1u << index;
        }
    }
    uint32_t m_events;
};

// parameters: output barycentric coordinates
template <int K, class callback = SolverEmptyCallback >
inline bool solveConstraints(float parameters[K * 2], float3 p_beg, float3 p_end, minimum_lbvh::Triangle tris[K], TriangleAttrib attribs[K], float eta, EventDescriptor eDescriptor, int maxIterations, float costTolerance, callback end_of_iter = SolverEmptyCallback())
{
    const int nParameters = K * 2;
    for (int i = 0; i < nParameters; i++)
    {
        parameters[i] = 1.0f / 3.0f;
    }

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
                    std::swap(wi, wo);
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

        sen::Mat<K * 2, 1> dparams = sen::solve_qr_overdetermined(A, b);
        for (int i = 0; i < nParameters; i++)
        {
            parameters[i] = parameters[i] - dparams(i, 0);
        }

        end_of_iter(iter, iter == maxIterations - 1);
    }
    return false;
}

inline bool occluded(
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

    float eps = 1.0e-6f;
    float3 from_safe = from + from_n * eps;
    float3 to_safe = to + to_n * eps;

    float3 rd = to_safe - from_safe;

    minimum_lbvh::Hit hit;
    hit.t = 1.0f;
    minimum_lbvh::intersect_stackfree(&hit, nodes, triangles, node, from_safe, rd, minimum_lbvh::invRd(rd), minimum_lbvh::RAY_QUERY_ANY);
    return hit.t < 1.0f;
}

template <int K>
inline bool contributablePath(float parameters[K * 2], float3 p_beg, float3 p_end, minimum_lbvh::Triangle tris[K], TriangleAttrib attribs[K], EventDescriptor eDescriptor, const minimum_lbvh::InternalNode* nodes, const minimum_lbvh::Triangle* sceneTriangles, minimum_lbvh::NodeIndex node )
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
            return false;
        }

        minimum_lbvh::Triangle tri = tris[k];

        float3 e0 = tri.vs[1] - tri.vs[0];
        float3 e1 = tri.vs[2] - tri.vs[0];
        vertices[k + 1] = tri.vs[0] + e0 * param_u + e1 * param_v;

        TriangleAttrib attrib = attribs[k];
        float3 ne0 = attrib.shadingNormals[1] - attrib.shadingNormals[0];
        float3 ne1 = attrib.shadingNormals[2] - attrib.shadingNormals[0];
        shadingNormals[k + 1] = normalize(attrib.shadingNormals[0] + ne0 * param_u + ne1 * param_v);
    }

    shadingNormals[0] = normalize(vertices[1] - vertices[0]);
    shadingNormals[K + 1] = normalize(vertices[K] - vertices[K + 1]);

    for (int k = 0; k < K; k++)
    {
        float3 wi = vertices[k] - vertices[k + 1];
        float3 wo = vertices[k + 2] - vertices[k + 1];
        float3 n = shadingNormals[k + 1];

        Event e = 0.0f < dot(wi, n) * dot(wo, n) ? Event::R : Event::T;
        if (eDescriptor.get(k) != e)
        {
            return false; // invalid event
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
            return false;
        }
    }
    return true;
}

inline float dAdw(float3 ro, float3 rd, float3 p_end, minimum_lbvh::Triangle* tris, TriangleAttrib* attribs, int nEvent)
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
            + saka::make_dval3(rd + T0) * differentials[0]
            + saka::make_dval3(rd + T1) * differentials[1];

        for (int j = 0; j < nEvent; j++)
        {
            minimum_lbvh::Triangle tri = tris[j];
            TriangleAttrib attrib = attribs[j];

            float3 ng = minimum_lbvh::normalOf(tri);

            saka::dval3 p = intersect_p_ray_plane(ro_j, rd_j, saka::make_dval3(ng), saka::make_dval3(tri.vs[0]));

            saka::dval u = dot(p - saka::make_dval3(tri.vs[1]), saka::make_dval3(cross(ng, tri.vs[1] - tri.vs[0]))) * 0.5f;
            saka::dval w = dot(p - saka::make_dval3(tri.vs[2]), saka::make_dval3(cross(ng, tri.vs[2] - tri.vs[1]))) * 0.5f;
            saka::dval v = dot(p - saka::make_dval3(tri.vs[0]), saka::make_dval3(cross(ng, tri.vs[0] - tri.vs[2]))) * 0.5f;

            saka::dval3 n =
                saka::make_dval3(attrib.shadingNormals[0]) * w +
                saka::make_dval3(attrib.shadingNormals[1]) * u +
                saka::make_dval3(attrib.shadingNormals[2]) * v;

            saka::dval3 wi = -rd_j;
            saka::dval3 wo = saka::reflection(wi, n);

            ro_j = p;
            rd_j = wo;
        }

        float3 p_last = { ro_j.x.v, ro_j.y.v, ro_j.z.v };
        saka::dval3 p_final = intersect_p_ray_plane(ro_j, rd_j, saka::make_dval3(p_last - p_end), saka::make_dval3(p_end));
        dAxis[i] = { p_final.x.g,  p_final.y.g,  p_final.z.g };

        //printf("x %.5f %.5f\n", p_final.x.v, p_light.x);
        //printf("y %.5f %.5f\n", p_final.y.v, p_light.y);
        //printf("z %.5f %.5f\n", p_final.z.v, p_light.z);
    }
    float3 crs = cross(dAxis[0], dAxis[1]);
    float dAdwValue = sqrtf(fmaxf(dot(crs, crs), 1.0e-15f));
    return dAdwValue;
}

struct InternalNormalBound
{
    minimum_lbvh::AABB normalBounds[2];
    uint32_t counter;
};

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

template <int K, class callback>
inline void traverseAdmissibleNodes(EventDescriptor admissibleEvents, float3 p_beg, float3 p_end, minimum_lbvh::InternalNode *internals, InternalNormalBound* internalsNormalBound, minimum_lbvh::Triangle* tris, TriangleAttrib* attribs, minimum_lbvh::NodeIndex node, callback admissibles )
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
            admissibles(admissibleTriangles);
        }
        else
        {
            for (int i = 0; i < 2; i++)
            {
                interval::intr3 triangle_intr = toIntr3(internals[currentNode.nodes[cutIndex].m_index].aabbs[i]);
                interval::intr3 prev_vert, next_vert;
                if (cutIndex == 0)
                {
                    prev_vert = p_beg_intr;
                }
                else
                {
                    prev_vert = toIntr3(nodeAABB(internals[currentNode.nodes[cutIndex - 1].m_index]));
                }

                if (cutIndex == K - 1)
                {
                    next_vert = p_end_intr;
                }
                else
                {
                    next_vert = toIntr3(nodeAABB(internals[currentNode.nodes[cutIndex + 1].m_index]));
                }

                interval::intr3 wi_intr = next_vert - triangle_intr;
                interval::intr3 wo_intr = prev_vert - triangle_intr;
                interval::intr3 normal_intr = toIntr3(internalsNormalBound[currentNode.nodes[cutIndex].m_index].normalBounds[i]);
                interval::intr3 R = interval::reflection(wi_intr, normal_intr);
                interval::intr3 Rc = interval::cross(R, wo_intr);

                if (interval::zeroIncluded(Rc))
                {
                    AdmissibleNodes<K> newNodes = currentNode;
                    newNodes.nodes[cutIndex] = internals[currentNode.nodes[cutIndex].m_index].children[i];
                    stack.push(newNodes);
                }
                
                // admissibleEvents

            }
        }

        //if (currentNode.m_isLeaf)
        //{
        //    AdmissibleTriangles<K> admissibleTriangles;
        //    admissibleTriangles.indices[0] = currentNode.m_index;
        //    admissibles(admissibleTriangles);
        //}
        //else
        {

            //for (int i = 0; i < 2; i++)
            //{
            //    interval::intr3 triangle_intr = toIntr3(internals[currentNode.m_index].aabbs[i]);
            //    interval::intr3 wi_intr = p_to - triangle_intr;
            //    interval::intr3 wo_intr = p_from - triangle_intr;
            //    interval::intr3 normal_intr = toIntr3(internalsNormalBound[currentNode.m_index].normalBounds[i]);
            //    interval::intr3 R = interval::reflection(wi_intr, normal_intr);
            //    interval::intr3 c = interval::cross(R, wo_intr);

            //    if (interval::zeroIncluded(c))
            //    {
            //        stack.push(internals[currentNode.m_index].children[i]);
            //    }
            //}
        }

        currentNode = stack.top(); stack.pop();
    }
}