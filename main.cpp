#include "pr.hpp"
#include <iostream>
#include <memory>
#include <stack>

#include "interval.h"
#include "helper_math.h"
#include "minimum_lbvh.h"
#include "saka.h"
#include "sen.h"
#include "sobol.h"
#include "pk.h"

inline void clamp_uv(float* u_inout, float* v_inout, minimum_lbvh::Triangle tri)
{
    float u = *u_inout;
    float v = *v_inout;
    float3 e0 = tri.vs[1] - tri.vs[0];
    float3 e1 = tri.vs[2] - tri.vs[0];
    float3 p = tri.vs[0] + e0 * u + e1 * v;
    if (v < 0.0f)
    {
        u = dot(e0, p - tri.vs[0]) / dot(e0, e0);
        u = clamp(u, 0.0f, 1.0f);
        v = 0.0f;
    }
    if (u < 0.0f)
    {
        v = dot(e1, p - tri.vs[0]) / dot(e1, e1);
        v = clamp(v, 0.0f, 1.0f);
        u = 0.0f;
    }
    if (1.0f < u + v)
    {
        float t = 1.0 / (u + v);
        u *= t;
        v *= t;
    }
    *u_inout = u;
    *v_inout = v;
}

// TODO unit test

inline glm::vec3 to(float3 p)
{
    return { p.x, p.y, p.z };
}
inline float3 to(glm::vec3 p)
{
    return { p.x, p.y, p.z };
}

inline void DrawAABB(interval::intr3 bound, glm::u8vec3 c, float lineWidth)
{
    pr::DrawAABB({ bound.x.l, bound.y.l, bound.z.l }, { bound.x.u, bound.y.u, bound.z.u }, c, lineWidth);
}

inline float3 mirror(float3 x, float3 n, float3 v0)
{
    return x + 2.0f * n * dot(n, v0 - x) / dot(n, n);
}

// The result vector is not normalized
inline float3 refraction(float3 wi /* normalized */, float3 n /* normalized */, float eta /* = eta_t / eta_i */)
{
    float3 crs = cross(wi, n);
    float k = eta * eta - dot(crs, crs);
    return -wi + (dot(wi, n) - sqrtf(k)) * n;
}
//inline float3 refraction2(float3 wi, float3 n /* normalized */, float eta /* = eta_t / eta_i */)
//{
//    float3 crs = cross(wi, n);
//    float k = dot(wi, wi) * eta * eta - dot(crs, crs);
//    return -wi + (dot(wi, n) - sqrtf(k)) * n;
//}
//inline float3 refraction3(float3 wi, float3 n /* normalized */, float eta /* = eta_t / eta_i */)
//{
//    float WIoN = dot(wi, n);
//    float k = dot(wi, wi) * ( eta * eta - 1.0f ) + WIoN * WIoN;
//    return -wi + (WIoN - sqrtf(k)) * n;
//}


inline float3 refraction_normal(float3 wi /*normalized*/, float3 wo /*normallized*/, float eta /* = eta_t / eta_i */)
{
    return -(wi + wo * eta);
}
inline saka::dval3 refraction_normal(saka::dval3 wi /*normalized*/, saka::dval3 wo /*normallized*/, float eta /* = eta_t / eta_i */)
{
    return -(wi + wo * eta);
}



//inline saka::dval3 refraction2(saka::dval3 wi, saka::dval3 n /* normalized */, float eta /* = eta_t / eta_i */)
//{
//    saka::dval3 crs = cross(wi, n);
//    saka::dval k = dot(wi, wi) * eta * eta - dot(crs, crs);
//    return -wi + n * (dot(wi, n) - saka::sqrt(k));
//}
//
//inline saka::dval3 refraction3(saka::dval3 wi, saka::dval3 n /* normalized */, float eta /* = eta_t / eta_i */)
//{
//    saka::dval WIoN = dot(wi, n);
//    saka::dval k = dot(wi, wi) * (eta * eta - 1.0f) + WIoN * WIoN;
//    return -wi + n * (WIoN - saka::sqrt(k));
//}
inline saka::dval3 refraction_norm_free(saka::dval3 wi, saka::dval3 n, float eta /* = eta_t / eta_i */)
{
    saka::dval NoN = dot(n, n);
    saka::dval WIoN = dot(wi, n);
    saka::dval WoW = dot(wi, wi);
    saka::dval k = NoN * WoW * (eta * eta - 1.0f) + WIoN * WIoN;
    return -wi * NoN + n * (WIoN - sqrt(k));
}

inline interval::intr3 toIntr3(minimum_lbvh::AABB aabb)
{
    return {
        {aabb.lower.x, aabb.upper.x},
        {aabb.lower.y, aabb.upper.y},
        {aabb.lower.z, aabb.upper.z}
    };
}


const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.99f;

struct Adam
{
    float m_m;
    float m_v;

    float optimize(float value, float g, float alpha, float beta1t, float beta2t)
    {
        float s = alpha;
        float m = ADAM_BETA1 * m_m + (1.0f - ADAM_BETA1) * g;
        float v = ADAM_BETA2 * m_v + (1.0f - ADAM_BETA2) * g * g;
        m_m = m;
        m_v = v;
        float m_hat = m / (1.0f - beta1t);
        float v_hat = v / (1.0f - beta2t);

        const float ADAM_E = 1.0e-15f;
        return value - s * m_hat / (sqrt(v_hat) + ADAM_E);
    }
};

inline float2 square2triangle(float2 square)
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

struct PolygonSoup
{
    minimum_lbvh::BVHCPUBuilder builder;
    std::vector<minimum_lbvh::Triangle> triangles;
    std::vector<TriangleAttrib> triangleAttribs;
};

struct InternalNormalBound
{
    minimum_lbvh::AABB normalBounds[2];
    uint32_t counter;
};
struct MirrorPolygonSoup
{
    minimum_lbvh::BVHCPUBuilder builder;
    std::vector<InternalNormalBound> internalsNormalBound;
    std::vector<minimum_lbvh::Triangle> triangles;
    std::vector<TriangleAttrib> triangleAttribs;
};

inline bool occluded(
    const minimum_lbvh::InternalNode* nodes,
    const minimum_lbvh::Triangle* triangles,
    minimum_lbvh::NodeIndex node,
    float3 from,
    float3 from_n,
    float3 to,
    float3 to_n )
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

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 2, 2, 2 };
    camera.lookat = { 0, 0, 0 };

    double e = GetElapsedTime();

    ITexture* texture = CreateTexture();

    SetDataDir(ExecutableDir());
    AbcArchive archive;
    std::string err;
    archive.open(GetDataPath("assets/scene.abc"), err);
    std::shared_ptr<FScene> scene = archive.readFlat(0, err);

    // all included
    PolygonSoup polygonSoup;

    // mirror 
    MirrorPolygonSoup mirrorPolygonSoup;

    scene->visitPolyMesh([&](std::shared_ptr<const FPolyMeshEntity> polymesh) {
        if (polymesh->visible() == false)
        {
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

            polygonSoup.triangles.push_back(tri);
            polygonSoup.triangleAttribs.push_back(attrib);
            indexBase += nVerts;
        }
    });

    {
        polygonSoup.builder.build(polygonSoup.triangles.data(), polygonSoup.triangles.size(), minimum_lbvh::BUILD_OPTION_CPU_PARALLEL);

        for (int i = 0; i < polygonSoup.triangles.size(); i++)
        {
            if (polygonSoup.triangleAttribs[i].material == Material::Mirror)
            {
                mirrorPolygonSoup.triangles.push_back(polygonSoup.triangles[i]);
                mirrorPolygonSoup.triangleAttribs.push_back(polygonSoup.triangleAttribs[i]);
            }
        }

        mirrorPolygonSoup.builder.build(mirrorPolygonSoup.triangles.data(), mirrorPolygonSoup.triangles.size(), minimum_lbvh::BUILD_OPTION_USE_NORMAL );
        mirrorPolygonSoup.internalsNormalBound.resize(polygonSoup.triangles.size() - 1);

        minimum_lbvh::InternalNode* internals = mirrorPolygonSoup.builder.m_internals.data();
        InternalNormalBound* internalsNormalBound = mirrorPolygonSoup.internalsNormalBound.data();
        for (uint32_t i = 0; i < mirrorPolygonSoup.builder.m_internals.size(); i++)
        {
            for (int j = 0; j < 2; j++)
            {
                minimum_lbvh::NodeIndex me = internals[i].children[j];
                if (!me.m_isLeaf)
                {
                    continue;
                }

                minimum_lbvh::NodeIndex parent(i, false);

                minimum_lbvh::Triangle tri = mirrorPolygonSoup.triangles[me.m_index];



                float3 normal = minimum_lbvh::normalOf(tri);
                minimum_lbvh::AABB normalBound;
                normalBound.setEmpty();

                // take shading normal into account
                const TriangleAttrib& attrib = mirrorPolygonSoup.triangleAttribs[me.m_index];
                for (int k = 0; k < 3; k++)
                {
                    normalBound.extend(attrib.shadingNormals[k]);
                }

                while (parent != minimum_lbvh::NodeIndex::invalid())
                {
                    int childIndex = internals[parent.m_index].children[0] == me ? 0 : 1;
                    internalsNormalBound[parent.m_index].normalBounds[childIndex] = normalBound;

                    uint32_t vindex = internalsNormalBound[parent.m_index].counter++;

                    if (vindex == 0)
                    {
                        break;
                    }

                    normalBound.extend(internalsNormalBound[parent.m_index].normalBounds[childIndex ^ 0x1]);

                    me = parent;
                    parent = internals[me.m_index].parent;
                }
            }
        }
    }

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        // ClearBackground(0.1f, 0.1f, 0.1f, 1);
        ClearBackground(texture);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

#if 0
        static glm::vec3 P0 = { 0, 1, 1 };
        ManipulatePosition(camera, &P0, 0.3f);

        static glm::vec3 P2 = { 0, 1, -1 };
        ManipulatePosition(camera, &P2, 0.3f);

        DrawText(P0, "P0");
        DrawText(P2, "P2");

        static float3 vs[3] = {
            {0.3f, 0.0f, 0.0f},
            {-0.5f, 0.0f, 0.1f},
            {0.1f, 0.0f, 0.6f},
        };

        for (int i = 0; i < 3; i++)
        {
            ManipulatePosition(camera, (glm::vec3*)&vs[i], 0.3f);
        }
        for (int i = 0; i < 3; i++)
        {
            DrawLine(to(vs[i]), to(vs[(i + 1) % 3]), { 255, 255, 255 }, 3);
        }
        
        interval::intr3 triangle =
            interval::make_intr3(vs[0].x, vs[0].y, vs[0].z) |
            interval::make_intr3(vs[1].x, vs[1].y, vs[1].z) |
            interval::make_intr3(vs[2].x, vs[2].y, vs[2].z);

        float3 normal = normalize(cross(vs[1] - vs[0], vs[2] - vs[0]));

        interval::intr3 n = interval::make_intr3(0.0f, 1.0f, 0.0f);
        interval::intr3 wi = interval::normalize( interval::make_intr3(P0.x, P0.y, P0.z) - triangle );
        interval::intr3 wo = interval::normalize( interval::make_intr3(P2.x, P2.y, P2.z) - triangle );

        interval::intr3 H = interval::normalize(wi + wo);

        float3 m = mirror(to(P2), normal, vs[0]);
        DrawSphere(to(m), 0.01f, { 0, 0, 255 });

        float3 hitP = {};
        {
            float3 rd = make_float3(P0.x, P0.y, P0.z) - m;
            float t;
            float u, v;
            float3 ng;
            if (minimum_lbvh::intersectRayTriangle(&t, &u, &v, &ng, 0.0f, MINIMUM_LBVH_FLT_MAX, m, rd, vs[0], vs[1], vs[2]))
            {
                hitP = m + t * rd;
                DrawLine(P0, to(hitP), { 255, 0, 0 }, 3);
                DrawLine(P2, to(hitP), { 255, 0, 0 }, 3);
            }
        }

        if (interval::intersects(H, interval::make_intr3(normal.x, normal.y, normal.z), 0.01f /* eps */) || 
            interval::intersects(-H, interval::make_intr3(normal.x, normal.y, normal.z), 0.01f /* eps */))
        {
            DrawSphere(to(hitP), 0.04f, {255, 0, 0});
        }
        else
        {
            DrawSphere(to(hitP), 0.01f, { 64, 64, 64 });
        }
        // interval::intr3 wo = interval::reflection(wi, n);

        // DrawAABB(wo, { 255, 255, 255 }, 1);
        DrawAABB(H, {255, 255, 255}, 1);
#endif


#if 0
        static float3 vs[3] = {
            {1.3f, 1.0f, 0.0f},
            {0.7f, 2.0f, -0.3f},
            {1.1f, 1.0f, 0.6f},
        };

        for (int i = 0; i < 3; i++)
        {
            ManipulatePosition(camera, (glm::vec3 *)&vs[i], 0.3f);
        }

        for (int i = 0; i < 3; i++)
        {
            DrawLine(to(vs[i]), to(vs[(i + 1) % 3]), { 64, 64, 64 }, 3);
        }

        float3 lower = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
        float3 upper = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

        PCG rng;
        for (int i = 0; i < 10000; i++)
        {
            float u = rng.uniformf();
            float v = rng.uniformf();
            if (u + v > 1.0f)
                continue;

            float3 dir = vs[0] + (vs[1] - vs[0]) * u + (vs[2] - vs[0]) * v;

            dir = normalize(dir);

            //DrawPoint(to(dir), {255, 255, 0}, 1);

            lower = fminf(lower, dir);
            upper = fmaxf(upper, dir);
        }

        //DrawAABB(to(lower), to(upper), { 255, 255, 255 }, 3);

        interval::intr3 pointOnTriangle =
            interval::make_intr3(vs[0].x, vs[0].y, vs[0].z) |
            interval::make_intr3(vs[1].x, vs[1].y, vs[1].z) |
            interval::make_intr3(vs[2].x, vs[2].y, vs[2].z);
        DrawAABB(pointOnTriangle, { 128, 128, 128 }, 2);


        interval::intr len = interval::sqrt(
            interval::square(pointOnTriangle.x) +
            interval::square(pointOnTriangle.y) +
            interval::square(pointOnTriangle.z)
        );
        interval::intr3 dir = pointOnTriangle / len;
        DrawAABB({ dir.x.l, dir.y.l, dir.z.l }, { dir.x.u, dir.y.u, dir.z.u }, { 255, 0, 0 }, 3);

        interval::intr3 normalized = interval::normalize(pointOnTriangle);
        DrawAABB({ normalized.x.l, normalized.y.l, normalized.z.l }, { normalized.x.u, normalized.y.u, normalized.z.u }, { 255, 255, 0 }, 3);

        // brute force
        {
            float3 lower = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
            float3 upper = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (int i = 0; i < 40000; i++)
            {
                float u = rng.uniformf();
                float v = rng.uniformf();
                float w = rng.uniformf();

                float3 randomP = {
                    lerp(pointOnTriangle.x.l, pointOnTriangle.x.u, u),
                    lerp(pointOnTriangle.y.l, pointOnTriangle.y.u, v),
                    lerp(pointOnTriangle.z.l, pointOnTriangle.z.u, w),
                };

                float3 dir = normalize(randomP);

                DrawPoint(to(dir), { 255, 255, 0 }, 1);

                lower = fminf(lower, dir);
                upper = fmaxf(upper, dir);
            }

            DrawAABB(to(lower), to(upper), { 0, 0, 255 }, 3);
        }

#endif

        // reflection
#if 0
        float margin = 0.1f;

        static glm::vec3 P0 = { 0, 1, 1 };
        ManipulatePosition(camera, &P0, 0.3f);

        static glm::vec3 N = { 0, 1.5f, 0 };
        ManipulatePosition(camera, &N, 0.3f);

        float3 wi = to(P0);

        DrawArrow({}, to(wi), 0.01f, { 255, 0, 0 });
        DrawArrow({}, N, 0.02f, { 0, 255, 255 });
        DrawText(to(wi), "wi");
        DrawText(N, "N");

        interval::intr3 wi_range = interval::relax(interval::make_intr3(wi.x, wi.y, wi.z), margin);

        DrawAABB(wi_range, { 255, 0, 0 }, 1);

        interval::intr3 wo_range = interval::reflection(wi_range, interval::make_intr3(N.x, N.y, N.z));
        DrawAABB(wo_range, { 0, 255, 0 }, 1);

        {
            PCG rng;

            float3 lower = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
            float3 upper = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (int i = 0; i < 10000; i++)
            {
                float3 wi_random = {
                    lerp(wi_range.x.l, wi_range.x.u, rng.uniformf()),
                    lerp(wi_range.y.l, wi_range.y.u, rng.uniformf()),
                    lerp(wi_range.z.l, wi_range.z.u, rng.uniformf()),
                };
                
                float3 wo_random = reflection(wi_random, to(N));

                DrawPoint(to(wo_random), { 255, 255, 0 }, 1);

                lower = fminf(lower, wo_random);
                upper = fmaxf(upper, wo_random);
            }

            DrawAABB(to(lower), to(upper), { 0, 0, 255 }, 3);
        }

#endif

#if 0
        // refraction
        float margin = 0.2f;
        float eta = 2.2f;

        static glm::vec3 P0 = { 0, 1, 1 };
        ManipulatePosition(camera, &P0, 0.3f);

        static glm::vec3 N = { 0, 1.f, 0 };
        ManipulatePosition(camera, &N, 0.3f);

        
        float3 wi_unnormalized = to(P0);

        DrawArrow({}, to(wi_unnormalized), 0.01f, { 255, 0, 0 });
        DrawArrow({}, N, 0.02f, { 0, 255, 255 });
        DrawText(to(wi_unnormalized), "wi(unnormalized)");
        DrawText(N, "N");

        interval::intr3 wi_range = interval::relax(interval::make_intr3(wi_unnormalized.x, wi_unnormalized.y, wi_unnormalized.z), margin);
        interval::intr3 wo_range = interval::refraction_norm_free(wi_range, interval::make_intr3(N), eta);

        DrawAABB(wi_range, { 255, 0, 0 }, 1);
        DrawAABB(wo_range, { 0, 255, 0 }, 1);

        //{
        //    interval::intr NoN = interval::lengthSquared(interval::make_intr3(N));
        //    interval::intr WIoN = interval::dot(wi_range, interval::make_intr3(N));
        //    interval::intr WIoWI = lengthSquared(wi_range);

        //    interval::intr alpha = WIoN;
        //    interval::intr beta = NoN * WIoWI;
        //    DrawAABB(interval::make_intr3(alpha, beta, 0), { 255, 0, 0 }, 1);
        //}

        // float3 wo = normalize(refraction(wi, normalize(to(N)), eta));
        float3 wo = normalize(refraction_norm_free(wi_unnormalized, to(N), eta) );
        
        DrawArrow({}, to(wo), 0.02f, { 255, 0, 255 });

        float3 wi = normalize(to(P0));
        //float3 ht = -(wi + wo * eta);
        float3 ht = refraction_normal(wi, wo, eta);

        DrawArrow({}, to(ht), 0.04f, { 255, 255, 255 });


        //if (1)
        //{
        //    DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });

        //    float eta = 1.96346688f;
        //    int N = 1000;
        //    for (int i = 0; i < N; i++)
        //    {
        //        glm::vec3 n = { 0, 3, 0 };
        //        glm::vec3 wi = glm::angleAxis( glm::pi<float>() * 0.5f * i / N, glm::vec3(0.0f, 0.0f, 1.0f)) * n * 0.4f;

        //        float R = fresnel_exact_norm_free(to(wi), to(n), eta);
        //        float T = 1.0f - R;
        //        float x = (float)i / N;
        //        DrawPoint({ x, R, 0 }, { 255, 255, 255 }, 3);

        //        float3 wo = refraction_norm_free(to(wi), to(n), eta);

        //        float T_inv = 1.0f - fresnel_exact_norm_free(wo, to(-n), 1.0f / eta);
        //        // printf("%f %f\n", T, T_inv);
        //    }
        //}

        if(0)
        {
            PCG rng;

            float3 lower = { +FLT_MAX, +FLT_MAX, +FLT_MAX };
            float3 upper = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (int i = 0; i < 40000; i++)
            {
                float3 wi_random = {
                    lerp(wi_range.x.l, wi_range.x.u, rng.uniformf()),
                    lerp(wi_range.y.l, wi_range.y.u, rng.uniformf()),
                    lerp(wi_range.z.l, wi_range.z.u, rng.uniformf()),
                };

                float3 wo_random = refraction_norm_free(wi_random, to(N), eta);
                DrawPoint(to(wo_random), { 255, 255, 0 }, 1);

                lower = fminf(lower, wo_random);
                upper = fmaxf(upper, wo_random);


                //float alpha = dot(wi_random, to(N));
                //float beta = dot(N, N) * dot(wi_random, wi_random);
                //DrawPoint({ alpha, beta, 0 }, { 255, 0, 255 }, 1);
            }

            DrawAABB(to(lower), to(upper), { 0, 0, 255 }, 3);
        }
#endif

#if 0
        // test with mesh
        pr::PrimBegin(pr::PrimitiveMode::Lines);

        for (auto tri : polygonSoup.triangles)
        {
            for (int j = 0; j < 3; ++j)
            {
                float3 v0 = tri.vs[j];
                float3 v1 = tri.vs[(j + 1) % 3];
                pr::PrimVertex(to(v0), { 255, 255, 255 });
                pr::PrimVertex(to(v1), { 255, 255, 255 });
            }
        }

        pr::PrimEnd();

        static glm::vec3 P0 = { 0, 0.5f, 0 };
        ManipulatePosition(camera, &P0, 0.3f);

        static glm::vec3 P2 = { -0.3f, -0.1f, 0.0f };
        ManipulatePosition(camera, &P2, 0.3f);

        DrawText(P0, "P0");
        DrawText(P2, "P2");

        // Brute force search 

        if(0)
        for (auto tri : mirrorPolygonSoup.triangles)
        {
            interval::intr3 triangle_intr =
                interval::make_intr3(tri.vs[0].x, tri.vs[0].y, tri.vs[0].z) |
                interval::make_intr3(tri.vs[1].x, tri.vs[1].y, tri.vs[1].z) |
                interval::make_intr3(tri.vs[2].x, tri.vs[2].y, tri.vs[2].z);

            interval::intr3 wi_intr = interval::normalize(interval::make_intr3(P0.x, P0.y, P0.z) - triangle_intr);
            interval::intr3 wo_intr = interval::normalize(interval::make_intr3(P2.x, P2.y, P2.z) - triangle_intr);
            interval::intr3 H_intr = interval::normalize(wi_intr + wo_intr);

            float3 normal = minimum_lbvh::normalOf(tri);

            float3 m = mirror(to(P2), normal, tri.vs[0]);
            float3 rd = make_float3(P0.x, P0.y, P0.z) - m;
            float t;
            float u, v;
            float3 ng;
            if (minimum_lbvh::intersectRayTriangle(&t, &u, &v, &ng, 0.0f, MINIMUM_LBVH_FLT_MAX, m, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
            {
                float3 hitP = m + t * rd;
                DrawLine(P0, to(hitP), { 255, 0, 0 }, 3);
                DrawLine(P2, to(hitP), { 255, 0, 0 }, 3);
            }

            if (interval::intersects(H_intr, interval::make_intr3(normal.x, normal.y, normal.z), 1.0e-8f /* eps */) ||
                interval::intersects(-H_intr, interval::make_intr3(normal.x, normal.y, normal.z), 1.0e-8f /* eps */))
            {
                for (int j = 0; j < 3; ++j)
                {
                    float3 v0 = tri.vs[j];
                    float3 v1 = tri.vs[(j + 1) % 3];
                    DrawLine(to(v0), to(v1), { 255, 255, 0 }, 3);
                }
            }
        }

        std::stack<minimum_lbvh::NodeIndex> stack;
        stack.push(minimum_lbvh::NodeIndex::invalid());
        minimum_lbvh::NodeIndex currentNode = mirrorPolygonSoup.builder.m_rootNode;

        if (1)
        while (currentNode != minimum_lbvh::NodeIndex::invalid())
        {
            if (currentNode.m_isLeaf)
            {
                minimum_lbvh::Triangle tri = mirrorPolygonSoup.triangles[currentNode.m_index];
                for (int j = 0; j < 3; ++j)
                {
                    float3 v0 = tri.vs[j];
                    float3 v1 = tri.vs[(j + 1) % 3];
                    DrawLine(to(v0), to(v1), { 255, 255, 0 }, 3);
                }
                float3 normal = minimum_lbvh::normalOf(tri);
                float3 m = mirror(to(P2), normal, tri.vs[0]);
                float3 rd = make_float3(P0.x, P0.y, P0.z) - m;
                float t;
                float u, v;
                float3 ng;
                if (minimum_lbvh::intersectRayTriangle(&t, &u, &v, &ng, 0.0f, MINIMUM_LBVH_FLT_MAX, m, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
                {
                    float3 hitP = m + t * rd;
                    DrawLine(P0, to(hitP), { 255, 0, 0 }, 3);
                    DrawLine(P2, to(hitP), { 255, 0, 0 }, 3);
                }
            }
            else
            {
                for (int i = 0; i < 2; i++)
                {
                    interval::intr3 triangle_intr = toIntr3(mirrorPolygonSoup.builder.m_internals[currentNode.m_index].aabbs[i]);

                    interval::intr3 wi_intr = interval::normalize(interval::make_intr3(P0.x, P0.y, P0.z) - triangle_intr);
                    interval::intr3 wo_intr = interval::normalize(interval::make_intr3(P2.x, P2.y, P2.z) - triangle_intr);

                    interval::intr3 h_intr = interval::normalize(wi_intr + wo_intr);
                    interval::intr3 normal_intr = toIntr3(mirrorPolygonSoup.internalsNormalBound[currentNode.m_index].normalBounds[i]);

                    if (interval::intersects(h_intr, normal_intr, 1.0e-8f /* eps */) ||
                        interval::intersects(-h_intr, normal_intr, 1.0e-8f /* eps */))
                    {
                        stack.push(mirrorPolygonSoup.builder.m_internals[currentNode.m_index].children[i]);
                    }
                }
            }

            currentNode = stack.top(); stack.pop();
        }
#endif 

#if 1
        //static float3 vs[3] = {
        //    {2.3f, 1.0f, -1.0f},
        //    
        //    {-0.539949894f, 1.0f, -0.342208207f },

        //    {1.1f, 1.0f, 1.6f},
        //};

        static minimum_lbvh::Triangle tri0 = {
            float3 {2.3f, 1.0f, -1.0f},
            float3 {1.1f, 1.0f, 1.6f},
            float3 {-0.539949894f, 1.0f, -0.342208207f },
        };

        for (int i = 0; i < 3; i++)
        {
            ManipulatePosition(camera, (glm::vec3*)&tri0.vs[i], 0.3f);
            DrawText(to(tri0.vs[i]), std::to_string(i));
        }
        for (int i = 0; i < 3; i++)
        {
            DrawLine(to(tri0.vs[i]), to(tri0.vs[(i + 1) % 3]), { 64, 64, 64 }, 3);
        }

        static glm::vec3 P0 = { 2.0f, 2.0f, 0 };
        // static glm::vec3 P0 = { 0.313918f, 1.19825f, -0.302908f };
        ManipulatePosition(camera, &P0, 0.3f);

        static glm::vec3 P2 = { -0.3f, -0.1f, 0.0f }; // refraction
        //  static glm::vec3 P2 = { -0.3f, 1.00517f, 0.0f }; // reflection
        ManipulatePosition(camera, &P2, 0.3f);

        DrawText(P0, "P0");
        DrawText(P2, "P2");

        // visualize cost function
        if (0)
        {
            PCG rng;

            float3 e0 = tri0.vs[1] - tri0.vs[0];
            float3 e1 = tri0.vs[2] - tri0.vs[0];
            for (int i = 0; i < 10000; i++)
            {
                float2 params = {};
                sobol::shuffled_scrambled_sobol_2d(&params.x, &params.y, i, 123, 456, 789);
                params = square2triangle(params);

                saka::dval3 P1 = saka::make_dval3(tri0.vs[0]) + saka::make_dval3(e0) * params.x + saka::make_dval3(e1) * params.y;
                //saka::dval3 wi = saka::normalize(saka::make_dval3(P0) - P1);
                //saka::dval3 wo = saka::normalize(saka::make_dval3(P2) - P1);
                saka::dval3 wi = saka::make_dval3(P0) - P1;
                saka::dval3 wo = saka::make_dval3(P2) - P1;

                saka::dval3 ht = refraction_normal(wi, wo, 1.3f);
                saka::dval3 n = -saka::make_dval3(minimum_lbvh::unnormalizedNormalOf(tri0));

                // refraction 
                //float eta = 1.3f;
                //saka::dval3 c = cross(n, wo * eta + wi);
                //float len2 = dot(c, c).v / 16;
                //glm::vec3 color = viridis(len2);

                // reflection
                //saka::dval3 c = cross(n, wo + wi);
                //float len2 = dot(c, c).v / 8;
                //glm::vec3 color = viridis(len2);

                //saka::dval3 R = reflection(wi, n);
                //saka::dval3 c = cross(R, wo);
                //float len2 = dot(c, c).v / 16;
                //glm::vec3 color = viridis(len2);

                saka::dval3 R = refraction_norm_free(wi, n, 1.3f);
                saka::dval3 c = cross(R, wo);
                float len2 = dot(c, c).v / 65536;
                glm::vec3 color = viridis(len2);

                float r = color.x * 255.0f;
                float g = color.y * 255.0f;
                float b = color.z * 255.0f;
                DrawPoint({ P1.x.v, P1.y.v, P1.z.v }, { r, g, b }, 3);

                float nL = len2 * 0.8f;
                DrawPoint({ P1.x.v + n.x.v * nL, P1.y.v + n.y.v * nL, P1.z.v + n.z.v * nL }, { r, g, b }, 3);
            }
        }

        static float param_a_init = 0.3f;
        static float param_b_init = 0.3f;
        if(0)
        {
            float param_a = param_a_init;
            float param_b = param_b_init;

            float beta1t = 1.0f;
            float beta2t = 1.0f;
            Adam adams[2] = {};

            int N_iter = 100;
            for (int iter = 0; iter < N_iter; iter++)
            {
                float3 e0 = tri0.vs[1] - tri0.vs[0];
                float3 e1 = tri0.vs[2] - tri0.vs[0];
                float3 P1 = tri0.vs[0] + e0 * param_a + e1 * param_b;

                if (iter == N_iter - 1)
                {
                    DrawLine(P0, to(P1), { 255, 0, 0 }, 2);
                    DrawLine(to(P1), P2, { 255, 0, 0 }, 2);
                }
                else
                {
                    DrawLine(P0, to(P1), { 64, 64, 64 }, 1);
                    DrawLine(to(P1), P2, { 64, 64, 64 }, 1);
                }
                DrawPoint(to(P1), { 255, 255, 255 }, 4);

                float dparams[2] = {};

                for (int i = 0; i < 2; i++)
                {
                    saka::dval params[2] = { param_a, param_b };
                    params[i].requires_grad();

                    saka::dval3 P1 = saka::make_dval3(tri0.vs[0]) + saka::make_dval3(e0) * params[0] + saka::make_dval3(e1) * params[1];
                    //saka::dval3 wi = saka::normalize(saka::make_dval3(P0) - P1);
                    //saka::dval3 wo = saka::normalize(saka::make_dval3(P2) - P1);
                    //saka::dval3 n = saka::make_dval3(minimum_lbvh::normalOf(tri0));

                    //saka::dval3 ht = refraction_normal(wi, wo, 1.3f);
                    //saka::dval3 c = saka::cross(n, ht);

                    //saka::dval3 c = cross(n, wo * 1.3f + wi);

                    saka::dval3 wi = saka::make_dval3(P0) - P1;
                    saka::dval3 wo = saka::make_dval3(P2) - P1;
                    saka::dval3 n = saka::make_dval3(-minimum_lbvh::unnormalizedNormalOf(tri0));
                    saka::dval3 c = saka::cross(wo, refraction_norm_free(wi, n, 1.3f));

                    dparams[i] = dot(c, c).g;
                }

                //float lr = 0.004f;
                //param_a = param_a - dparams[0] * lr;
                //param_b = param_b - dparams[1] * lr;
                float lr = 0.02f;
                beta1t *= ADAM_BETA1;
                beta2t *= ADAM_BETA2;
                param_a = adams[0].optimize(param_a, dparams[0], lr, beta1t, beta2t);
                param_b = adams[1].optimize(param_b, dparams[1], lr, beta1t, beta2t);
            }
        }

        // Single Event
        if(0)
        {
            float param_a = param_a_init;
            float param_b = param_b_init;

            float curCost = 10.0f;
            float alpha = 1.0f;

            int N_iter = 300;
            for (int iter = 0; iter < N_iter; iter++)
            {
                float3 e0 = tri0.vs[1] - tri0.vs[0];
                float3 e1 = tri0.vs[2] - tri0.vs[0];
                float3 P1 = tri0.vs[0] + e0 * param_a + e1 * param_b;

                if (iter == N_iter - 1)
                {
                    DrawLine(P0, to(P1), { 255, 0, 0 }, 2);
                    DrawLine(to(P1), P2, { 255, 0, 0 }, 2);
                }
                else
                {
                    DrawLine(P0, to(P1), { 64, 64, 64 }, 1);
                    DrawLine(to(P1), P2, { 64, 64, 64 }, 1);
                }

                DrawPoint(to(P1), { 255, 255, 255 }, 4);
                DrawText(to(P1), std::to_string(iter));

                sen::Mat<3, 2> A;
                sen::Mat<3, 1> b;

                float newCost;
                for (int i = 0; i < 2; i++)
                {
                    saka::dval params[2] = { param_a, param_b };
                    params[i].requires_grad();

                    saka::dval3 P1 = saka::make_dval3(tri0.vs[0]) + saka::make_dval3(e0) * params[0] + saka::make_dval3(e1) * params[1];
                    //saka::dval3 wi = saka::normalize(saka::make_dval3(P0) - P1);
                    //saka::dval3 wo = saka::normalize(saka::make_dval3(P2) - P1);
                    //saka::dval3 wi = saka::make_dval3(P0) - P1;
                    //saka::dval3 wo = saka::make_dval3(P2) - P1;
                    //saka::dval3 n = -saka::make_dval3(minimum_lbvh::normalOf(tri0));

                    //saka::dval3 ht = refraction_normal(wi, wo, 1.3f);
                    //saka::dval3 c = cross(n, ht);

                    //saka::dval3 ht = wi + wo;
                    //saka::dval3 c = cross(n, ht);

                    // refraction
                    // saka::dval3 c = cross(n, wo * 1.3f + wi);

                    // reflection
                    //saka::dval3 c = cross(n, wo + wi);

                    //saka::dval3 R = reflection(wi, n);
                    //saka::dval3 c = cross(R, wo);

                    //saka::dval3 R = refraction2(wi, n, 1.3f);
                    //saka::dval3 c = cross(R, wo);

                    saka::dval3 wi = saka::make_dval3(P0) - P1;
                    saka::dval3 wo = saka::make_dval3(P2) - P1;
                    saka::dval3 n = saka::make_dval3(-minimum_lbvh::unnormalizedNormalOf(tri0));
                    saka::dval3 R = refraction_norm_free(wi, n, 1.3f);
                    saka::dval3 c = saka::cross(wo, R);

                    A(0, i) = c.x.g;
                    A(1, i) = c.y.g;
                    A(2, i) = c.z.g;

                    b(0, 0) = c.x.v;
                    b(1, 0) = c.y.v;
                    b(2, 0) = c.z.v;

                    newCost = dot(c, c).v;
                }
                
                sen::Mat<2, 1> dparams = sen::solve_qr_overdetermined(A, b);

                //if ( newCost < curCost )
                //{
                //    alpha = fminf(alpha * (4.0f / 3.0f), 1.0f);
                //}
                //else
                //{
                //    alpha = fmaxf(alpha * (1.0f / 3.0f), 1.0f / 32.0f);
                //}
                //curCost = newCost;
                //float movement = sqrtf(dparams(0, 0) * dparams(0, 0) + dparams(1, 0) * dparams(1, 0));
                //float maxStep = 0.25f;
                //float clampScale = fminf(1.0f, maxStep / movement);

                //param_a = param_a - alpha * dparams(0, 0) * clampScale;
                //param_b = param_b - alpha * dparams(1, 0) * clampScale;

                param_a = param_a - dparams(0, 0);
                param_b = param_b - dparams(1, 0);
            }
        }

        if (1)
        {
            // 2 Events
            static minimum_lbvh::Triangle tri1 = {
                float3 {2.3f, 1.5f, -1.0f},
                float3 {-0.539949894f, 1.5f, -0.342208207f },
                float3 {1.1f, 1.5f, 1.6f},
            };

            for (int i = 0; i < 3; i++)
            {
                ManipulatePosition(camera, (glm::vec3*)&tri1.vs[i], 0.3f);
                DrawText(to(tri1.vs[i]), std::to_string(i));
            }

            float3 n = minimum_lbvh::normalOf(tri1);

            for (int i = 0; i < 3; i++)
            {
                DrawLine(to(tri1.vs[i]), to(tri1.vs[(i + 1) % 3]), { 64, 64, 64 }, 3);
            }

            const int K = 2;
            minimum_lbvh::Triangle admissibleTriangles[K] = { tri1, tri0 };
            float indexOfRefractions[2] = {1.3f, 1.3f};

            float3 P_beg = to(P0);
            float3 P_end = to(P2);

            const int nParameters = K * 2;
            float parameters[nParameters];
            for (int i = 0; i < nParameters; i++)
            {
                parameters[i] = 1.0f / 3.0f;
            }

            float curCosts[K];
            float alphas[K];
            for (int k = 0; k < K; k++)
            {
                curCosts[k] = 1.0e+15f;
                alphas[k] = 1.0f;
            }

            int N_iter = 300;
            for (int iter = 0; iter < N_iter; iter++)
            {
                sen::Mat<K * 3, K * 2> A;
                sen::Mat<K * 3, 1> b;

                float newCosts[K];
                for (int k = 0; k < K; k++)
                {
                    newCosts[k] = 1.0e+15f;
                }

                for (int i = 0; i < nParameters; i++)
                {
                    saka::dval parameters_optimizable[nParameters];
                    for (int j = 0; j < nParameters; j++)
                    {
                        parameters_optimizable[j] = parameters[j];
                    }
                    parameters_optimizable[i].requires_grad();

                    saka::dval3 vertices[K + 2];
                    vertices[0]     = saka::make_dval3(P_beg);
                    vertices[K + 1] = saka::make_dval3(P_end);

                    for (int k = 0; k < K; k++)
                    {
                        saka::dval param_u = parameters_optimizable[k * 2 + 0];
                        saka::dval param_v = parameters_optimizable[k * 2 + 1];

                        minimum_lbvh::Triangle tri = admissibleTriangles[k];

                        float3 e0 = tri.vs[1] - tri.vs[0];
                        float3 e1 = tri.vs[2] - tri.vs[0];
                        vertices[k + 1] = saka::make_dval3(tri.vs[0]) + saka::make_dval3(e0) * param_u + saka::make_dval3(e1) * param_v;
                    }
                    // debug draw
                    if (i == 0)
                    {
                        for (int j = 0; j < K + 1; j++)
                        {

                            glm::vec3 a = { vertices[j].x.v, vertices[j].y.v, vertices[j].z.v };
                            glm::vec3 b = { vertices[j + 1].x.v, vertices[j + 1].y.v, vertices[j + 1].z.v };
                            if (iter == N_iter - 1)
                            {
                                DrawLine(a, b, { 255, 64, 64 }, 2);
                            }
                            else
                            {
                                DrawLine(a, b, { 64, 64, 64 }, 1);
                            }
                            DrawPoint(b, { 255, 255, 255 }, 4);
                            DrawText(b, std::to_string(iter));
                        }
                    }

                    for (int k = 0; k < K; k++)
                    {
                        saka::dval3 wi = vertices[k]     - vertices[k + 1];
                        saka::dval3 wo = vertices[k + 2] - vertices[k + 1];
                        saka::dval3 n = saka::make_dval3(minimum_lbvh::unnormalizedNormalOf(admissibleTriangles[k]));

                        float eta = indexOfRefractions[k]; // eta_o / ita_i
                        if (dot(wi, n).v < 0.0f)
                        {
                            std::swap(wi, wo);
                        }

                        // refraction
                        //saka::dval3 c = cross(n, wo * eta + wi);

                        saka::dval3 R = refraction_norm_free(wi, n, eta);
                        saka::dval3 c = saka::cross(wo, R);

                        // reflection
                        // saka::dval3 c = cross(n, wo + wi);

                        A(k * 3 + 0, i) = c.x.g;
                        A(k * 3 + 1, i) = c.y.g;
                        A(k * 3 + 2, i) = c.z.g;

                        b(k * 3 + 0, 0) = c.x.v;
                        b(k * 3 + 1, 0) = c.y.v;
                        b(k * 3 + 2, 0) = c.z.v;

                        newCosts[k] = dot(c, c).v;
                    }
                }

                sen::Mat<K * 2, 1> dparams = sen::solve_qr_overdetermined(A, b);
                for (int i = 0; i < nParameters; i++)
                {
                    parameters[i] = parameters[i] - dparams(i, 0);
                }
            }
        }
#endif

        // Rendering
#if 0
        float3 light_intencity = { 1.0f, 1.0f, 1.0f };
        static glm::vec3 p_light = { 0, 1, 1 };
        ManipulatePosition(camera, &p_light, 0.3f);
        DrawText(p_light, "light");

        int stride = 2;
        Image2DRGBA8 image;
        image.allocate(GetScreenWidth() / stride, GetScreenHeight() / stride);

        CameraRayGenerator rayGenerator(GetCurrentViewMatrix(), GetCurrentProjMatrix(), image.width(), image.height());

        //for (int j = 0; j < image.height(); ++j)
        ParallelFor(image.height(), [&](int j) {
            for (int i = 0; i < image.width(); ++i)
            {
                glm::vec3 ro, rd;
                rayGenerator.shoot(&ro, &rd, i, j, 0.5f, 0.5f);

                minimum_lbvh::Hit hit;
                minimum_lbvh::intersect_stackfree(&hit, polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode, to(ro), to(rd), minimum_lbvh::invRd(to(rd)));
                if (hit.t == MINIMUM_LBVH_FLT_MAX)
                {
                    image(i, j) = { 0, 0, 0, 255 };
                    continue;
                }
                
                float3 p = to(ro) + to(rd) * hit.t;
                float3 n = normalize(hit.ng);

                if (0.0f < dot(n, to(rd)))
                {
                    n = -n;
                }

                if (polygonSoup.triangleAttribs[hit.triangleIndex].material == Material::Mirror)
                {
                    // handle later
                    image(i, j) = { 0, 255, 255, 255 };
                    continue;
                }

                //float3 color = (n + make_float3(1.0f)) * 0.5f;
                //image(i, j) = { 255 * color.x, 255 * color.y, 255 * color.z, 255 };

                float3 toLight = to(p_light) - p;
                float d2 = dot(toLight, toLight);
                float3 reflectance = { 0.75f, 0.75f, 0.75f };
                
                float3 L = {};

                bool invisible = occluded(polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode, p, n, to(p_light), { 0, 0, 0 });
                if (!invisible)
                {
                    L += reflectance * light_intencity / d2 * fmaxf(dot(normalize(toLight), n), 0.0f);
                }

                interval::intr3 p_to   = interval::make_intr3(p_light.x, p_light.y, p_light.z);
                interval::intr3 p_from = interval::make_intr3(p.x, p.y, p.z);

                std::stack<minimum_lbvh::NodeIndex> stack;
                stack.push(minimum_lbvh::NodeIndex::invalid());
                minimum_lbvh::NodeIndex currentNode = mirrorPolygonSoup.builder.m_rootNode;

                while (currentNode != minimum_lbvh::NodeIndex::invalid())
                {
                    if (currentNode.m_isLeaf)
                    {
                        minimum_lbvh::Triangle tri = mirrorPolygonSoup.triangles[currentNode.m_index];

                        // float3 normal = minimum_lbvh::normalOf(tri);
                        TriangleAttrib attrib = mirrorPolygonSoup.triangleAttribs[currentNode.m_index];

                        float param_a = 1.0f / 3.0f;
                        float param_b = 1.0f / 3.0f;

                        float3 e0 = tri.vs[1] - tri.vs[0];
                        float3 e1 = tri.vs[2] - tri.vs[0];
                        float3 en0 = attrib.shadingNormals[1] - attrib.shadingNormals[0];
                        float3 en1 = attrib.shadingNormals[2] - attrib.shadingNormals[0];

                        bool converged = false;

                        int N_iter = 64;
                        for (int iter = 0; iter < N_iter; iter++)
                        {
                            float3 P1 = tri.vs[0] + e0 * param_a + e1 * param_b;

                            sen::Mat<3, 2> A;
                            sen::Mat<3, 1> b;

                            float newCost = 0.0f;

                            for (int i = 0; i < 2; i++)
                            {
                                saka::dval params[2] = { param_a, param_b };
                                params[i].requires_grad();

                                saka::dval3 P1 = saka::make_dval3(tri.vs[0]) + saka::make_dval3(e0) * params[0] + saka::make_dval3(e1) * params[1];
                                saka::dval3 wi = saka::make_dval3(p) - P1;
                                saka::dval3 wo = saka::make_dval3(p_light) - P1;
                                saka::dval3 n = saka::make_dval3(attrib.shadingNormals[0]) + saka::make_dval3(en0) * params[0] + saka::make_dval3(en1) * params[1];

                                saka::dval3 R = reflection(wi, n);
                                saka::dval3 c = cross(wo, R);

                                newCost = dot(c, c).v;

                                A(0, i) = c.x.g;
                                A(1, i) = c.y.g;
                                A(2, i) = c.z.g;

                                b(0, 0) = c.x.v;
                                b(1, 0) = c.y.v;
                                b(2, 0) = c.z.v;
                            }

                            if (newCost < 1.0e-10f)
                            {
                                converged = true;
                                break;
                            }

                            sen::Mat<2, 1> dparams = sen::solve_qr_overdetermined(A, b);
                            param_a = param_a - dparams(0, 0);
                            param_b = param_b - dparams(1, 0);
                        }

                        if(converged && 0.0f <= param_a && 0.0f <= param_b && param_a + param_b < 1.0f )
                        {
                            float3 hitP = tri.vs[0] + e0 * param_a + e1 * param_b;
                            float3 hitN = attrib.shadingNormals[0] + en0 * param_a + en1 * param_b;

                            bool invisible =
                                occluded(polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode, p, n, hitP, normalize(hitN)) ||
                                occluded(polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode, hitP, normalize(hitN), to(p_light), { 0, 0, 0 });

                            if (!invisible)
                            {
                                float dAdwValue = dAdw(p, hitP - p, to(p_light), &tri, &attrib, 1);

                                L += reflectance * light_intencity / dAdwValue * fmaxf(dot(normalize(hitP - p), n), 0.0f);

                                //float3 d0to1 = hitP - p;
                                //float3 d1to2 = to(p_light) - hitP;
                                //float d = sqrt(dot(d0to1, d0to1)) + sqrt(dot(d1to2, d1to2));
                                //L += reflectance * light_intencity / ( d * d ) * fmaxf(dot(normalize(hitP - p), n), 0.0f);
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 2; i++)
                        {
                            interval::intr3 triangle_intr = toIntr3(mirrorPolygonSoup.builder.m_internals[currentNode.m_index].aabbs[i]);
                            interval::intr3 wi_intr = p_to - triangle_intr;
                            interval::intr3 wo_intr = p_from - triangle_intr;
                            interval::intr3 normal_intr = toIntr3(mirrorPolygonSoup.internalsNormalBound[currentNode.m_index].normalBounds[i]);
                            interval::intr3 R = interval::reflection(wi_intr, normal_intr);
                            interval::intr3 c = interval::cross(R, wo_intr);

                            if (interval::zeroIncluded(c))
                            {
                                stack.push(mirrorPolygonSoup.builder.m_internals[currentNode.m_index].children[i]);
                            }
                        }
                    }

                    currentNode = stack.top(); stack.pop();
                }


                float3 color = clamp(L, 0.0f, 1.0f);
                image(i, j) = { 
                    255 * powf(color.x, 1.0f / 2.2f), 
                    255 * powf(color.y, 1.0f / 2.2f), 
                    255 * powf(color.z, 1.0f / 2.2f), 255};
            };
        }
        );

        texture->upload(image);

#endif

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        //ImGui::SliderFloat("param_a_init", &param_a_init, 0, 1);
        //ImGui::SliderFloat("param_b_init", &param_b_init, 0, 1);
        //if (ImGui::Button("restart"))
        //{
        //    param_a = 0.3f;
        //    param_b = 0.3f;
        //}

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
