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

bool g_bruteforce = false;

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

struct DeltaPolygonSoup
{
    minimum_lbvh::BVHCPUBuilder builder;
    std::vector<InternalNormalBound> internalsNormalBound;
    std::vector<minimum_lbvh::Triangle> triangles;
    std::vector<TriangleAttrib> triangleAttribs;
};

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 2, 2, -2 };
    camera.lookat = { 0, 0, 0 };

    double e = GetElapsedTime();

    ITexture* texture = CreateTexture();

    SetDataDir(ExecutableDir());
    AbcArchive archive;
    std::string err;
    archive.open(GetDataPath("assets/scene.abc"), err);
    std::shared_ptr<FScene> scene = archive.readFlat(0, err);

    int debug_index = 10;

    // all included
    PolygonSoup polygonSoup;

    // mirror 
    DeltaPolygonSoup deltaPolygonSoup;

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

            polygonSoup.triangles.push_back(tri);
            polygonSoup.triangleAttribs.push_back(attrib);
            indexBase += nVerts;
        }
    });

    {
        polygonSoup.builder.build(polygonSoup.triangles.data(), polygonSoup.triangles.size(), minimum_lbvh::BUILD_OPTION_CPU_PARALLEL);

        for (int i = 0; i < polygonSoup.triangles.size(); i++)
        {
            if (polygonSoup.triangleAttribs[i].material == Material::Mirror ||
                polygonSoup.triangleAttribs[i].material == Material::Dielectric)
            {
                deltaPolygonSoup.triangles.push_back(polygonSoup.triangles[i]);
                deltaPolygonSoup.triangleAttribs.push_back(polygonSoup.triangleAttribs[i]);
            }
        }

        deltaPolygonSoup.builder.build(deltaPolygonSoup.triangles.data(), deltaPolygonSoup.triangles.size(), minimum_lbvh::BUILD_OPTION_USE_NORMAL );
        deltaPolygonSoup.internalsNormalBound.resize(polygonSoup.triangles.size() - 1);

        minimum_lbvh::InternalNode* internals = deltaPolygonSoup.builder.m_internals.data();
        InternalNormalBound* internalsNormalBound = deltaPolygonSoup.internalsNormalBound.data();
        for (uint32_t i = 0; i < deltaPolygonSoup.builder.m_internals.size(); i++)
        {
            for (int j = 0; j < 2; j++)
            {
                minimum_lbvh::NodeIndex me = internals[i].children[j];
                if (!me.m_isLeaf)
                {
                    continue;
                }

                minimum_lbvh::NodeIndex parent(i, false);

                minimum_lbvh::Triangle tri = deltaPolygonSoup.triangles[me.m_index];



                float3 normal = minimum_lbvh::normalOf(tri);
                minimum_lbvh::AABB normalBound;
                normalBound.setEmpty();

                // take shading normal into account
                const TriangleAttrib& attrib = deltaPolygonSoup.triangleAttribs[me.m_index];
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
        float eta = 1.5f;

        static glm::vec3 P0 = { 0, 1, 1 };
        ManipulatePosition(camera, &P0, 0.3f);

        //static glm::vec3 P2 = { 0, -1, -1 };
        //ManipulatePosition(camera, &P2, 0.3f);

        static glm::vec3 N = { 0, 1.f, 0 };
        ManipulatePosition(camera, &N, 0.3f);

        
        float3 wi_unnormalized = to(P0);
        //float3 wo_unnormalized = to(P2);

        DrawArrow({}, to(wi_unnormalized), 0.01f, { 255, 0, 0 });
        // DrawArrow({}, to(wo_unnormalized), 0.01f, { 0, 0, 255 });

        DrawArrow({}, N, 0.02f, { 0, 255, 255 });
        DrawText(to(wi_unnormalized), "wi(unnormalized)");
        DrawText(N, "N");

        interval::intr3 wi_range = interval::relax(interval::make_intr3(wi_unnormalized.x, wi_unnormalized.y, wi_unnormalized.z), margin);
        interval::intr3 wo_range;
        
        DrawAABB(wi_range, { 255, 0, 0 }, 1);
        if (interval::refraction_norm_free(&wo_range, wi_range, interval::make_intr3(N), eta))
        {
            DrawAABB(wo_range, { 0, 255, 0 }, 1);
        }

        //interval::intr3 crs = cross(wo_range, interval::make_intr3(wo_unnormalized));
        //DrawAABB(crs, { 255, 255, 255 }, 1);

        //{
        //    interval::intr NoN = interval::lengthSquared(interval::make_intr3(N));
        //    interval::intr WIoN = interval::dot(wi_range, interval::make_intr3(N));
        //    interval::intr WIoWI = lengthSquared(wi_range);

        //    interval::intr alpha = WIoN;
        //    interval::intr beta = NoN * WIoWI;
        //    DrawAABB(interval::make_intr3(alpha, beta, 0), { 255, 0, 0 }, 1);
        //}

        // float3 wo = normalize(refraction(wi, normalize(to(N)), eta));
        float3 wo;
        if (refraction_norm_free(&wo, wi_unnormalized, to(N), eta))
        {
            wo = normalize(wo);
        
            DrawArrow({}, to(wo), 0.02f, { 255, 0, 255 });

            float3 wi = normalize(to(P0));
            //float3 ht = -(wi + wo * eta);
            float3 ht = refraction_normal(wi, wo, eta);

            DrawArrow({}, to(ht), 0.04f, { 255, 255, 255 });
        }

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

        if(1)
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

                float3 wo_random;
                if (refraction_norm_free(&wo_random, wi_random, to(N), eta))
                {
                    DrawPoint(to(wo_random), { 255, 255, 0 }, 1);

                    lower = fminf(lower, wo_random);
                    upper = fmaxf(upper, wo_random);
                }

                //float alpha = dot(wi_random, to(N));
                //float beta = dot(N, N) * dot(wi_random, wi_random);
                //DrawPoint({ alpha, beta, 0 }, { 255, 0, 255 }, 1);
            }

            DrawAABB(to(lower), to(upper), { 0, 0, 255 }, 3);
        }
#endif

#if 0
        float margin = 0.1f;
        float eta = 1.5f;

        static glm::vec3 P0 = { 0, 1, 1 };
        ManipulatePosition(camera, &P0, 0.3f);
        DrawText(P0, "wi");

        static glm::vec3 P2 = { 0, -1, -0.506008 };
        ManipulatePosition(camera, &P2, 0.3f);
        DrawText(P2, "wi");

        interval::intr3 wi = interval::relax(interval::make_intr3(P0), margin);
        interval::intr3 wo = interval::relax(interval::make_intr3(P2), margin);

        DrawAABB(wi, { 255, 0, 0 }, 2);
        DrawAABB(wo, { 0, 255, 0}, 2);

        interval::intr lenWi = sqrt(interval::lengthSquared(wi));
        interval::intr lenWo = sqrt(interval::lengthSquared(wo));
        
        //interval::intr3 ht = refraction_normal(wi, wo, wieta);
        interval::intr3 ht = -( wo * lenWi * eta + wi * lenWo );

        {
            interval::intr dx =
                -eta * sqrt(interval::lengthSquared(wi)) *
                sqrt(interval::lengthSquared(wo)) - wi.x * wo.x;

            interval::intr dy =
                -eta * sqrt(interval::lengthSquared(wi)) *
                sqrt(interval::lengthSquared(wo)) - wi.y * wo.y;

            interval::intr dz =
                -eta * sqrt(interval::lengthSquared(wi)) *
                sqrt(interval::lengthSquared(wo)) - wi.z * wo.z;

            interval::intr da =
                -sqrt(interval::lengthSquared(wi)) *
                sqrt(interval::lengthSquared(wo)) - wi.z * wo.z * eta;

            // float lx = wo.x.l * lenWi.u * eta + wi.x.l * lenWo.u;

            //float lx = 
            //    - wo.x.u * (0.0f < -wo.x.u ? lenWi.l : lenWi.u) * eta + (-wi.x).l * sqrt(
            //        wo.x.u * wo.x.u + 
            //        ( 0.0f < (-wi.x).l ?
            //        wo.y.l * wo.y.l + wo.z.l * wo.z.l:
            //        wo.y.u * wo.y.u + wo.z.u * wo.z.u )
            //    );
            //printf("%f %f\n", dx.l, dx.u);
            //printf("%f %f\n", dy.l, dy.u);
            //printf("%f %f\n", dz.l, dz.u);

            //{
            //    auto wo_fixed = wo;
            //    wo_fixed.x = wo.x.u;
            //    interval::intr3 ht_limited = -(wo_fixed * lenWi * eta + wi * sqrt(interval::lengthSquared(wo_fixed)));
            //    ht.x.l = ht_limited.x.l;
            //}

            //{
            //    auto wo_fixed = wo;
            //    wo_fixed.y = wo.y.u;
            //    interval::intr3 ht_limited = -(wo_fixed * lenWi * eta + wi * sqrt(interval::lengthSquared(wo_fixed)));
            //    ht.y.l = ht_limited.y.l;
            //}
            //{
            //    auto wi_fixed = wi;
            //    wi_fixed.z = wi.z.u;

            //    auto wo_fixed = wo;
            //    wo_fixed.z = wo.z.u;
            //    //interval::intr3 ht_limited = -(wo_fixed * lenWi * eta + wi * sqrt(interval::lengthSquared(wo_fixed)));
            //    interval::intr3 ht_limited = -(wo_fixed * sqrt(interval::lengthSquared(wi_fixed)) * eta + wi_fixed * sqrt(interval::lengthSquared(wo_fixed)));
            //    ht.z.l = ht_limited.z.l;
            //}

            interval::intr3 ht_tight = refraction_normal_tight(wi, wo, eta);
            DrawAABB(ht_tight, { 0, 0, 255 }, 2);
            printf("");
        }
        DrawAABB(ht, { 0, 255, 255 }, 2);

        PCG rng;
        auto random_of = [&rng](interval::intr3 r) -> float3 {
            return {
                lerp(r.x.l, r.x.u, rng.uniformf()),
                lerp(r.y.l, r.y.u, rng.uniformf()),
                lerp(r.z.l, r.z.u, rng.uniformf()),
            };
        };

        PrimBegin(PrimitiveMode::Points, 2);
        for (int i = 0; i < 100000; i++)
        {
            float3 wi_r = random_of(wi);
            float3 wo_r = random_of(wo);
            float lenWi = length(wi_r);
            float lenWo = length(wo_r);

            float3 ht = -(wo_r * lenWi * eta + wi_r * lenWo);
            
            float k = dot(ht, ht) * dot(wi_r, wi_r) * (eta * eta - 1.0f) + dot(ht, wi_r) * dot(ht, wi_r);
            //if (i == 0)
            //{
            //    printf("s %f\n", dot(ht, wi_r) * dot(ht, wo_r));
            //}

            if (dot(ht, wi_r) * dot(ht, wo_r) < 0.0f)
            {
                PrimVertex(to(ht), { 255, 0, 0 });
                // DrawPoint(to(ht), {255, 0, 0}, 3);
            }

        }
        PrimEnd();

#endif

#if 0
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

        minimum_lbvh::Triangle tri0 = deltaPolygonSoup.triangles[13];
        minimum_lbvh::Triangle tri1 = deltaPolygonSoup.triangles[91];
        for (int j = 0; j < 3; ++j)
        {
            float3 v0 = tri0.vs[j];
            float3 v1 = tri0.vs[(j + 1) % 3];
            DrawLine(to(v0), to(v1), { 255, 0, 0 }, 4);
        }
        for (int j = 0; j < 3; ++j)
        {
            float3 v0 = tri1.vs[j];
            float3 v1 = tri1.vs[(j + 1) % 3];
            DrawLine(to(v0), to(v1), { 0, 255, 0 }, 2);
        }

        minimum_lbvh::AABB bound0;
        bound0.setEmpty();
        for (float3 p : tri0.vs)
        {
            bound0.extend(p);
        }
        interval::intr3 p0_intr = toIntr3(bound0);

        minimum_lbvh::AABB bound1;
        bound1.setEmpty();
        for (float3 p : tri1.vs)
        {
            bound1.extend(p);
        }
        interval::intr3 p1_intr = toIntr3(bound1);

        interval::intr3 wo = p1_intr - p0_intr;

        DrawAABB(wo, {0, 255, 0}, 1);

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

        //for (int i = 0; i < polygonSoup.triangles.size(); i++)
        //{
        //    for (int j = 0; j < 3; j++)
        //    {
        //        float3 p = polygonSoup.triangles[i].vs[j];
        //        float3 n = polygonSoup.triangleAttribs[i].shadingNormals[j];

        //        DrawLine(to(p), to(p + n * 0.1f), { 255, 0, 255 });
        //    }
        //}

        //static glm::vec3 P0 = { 1, 0.5f, 0 };
        //static glm::vec3 P2 = { -0.3f, -0.1f, 0.0f };

        static glm::vec3 P0 = { -0.0187847f, -0.0271853f, 0.5f };
        static glm::vec3 P2 = { 0.0443912f, 0.344693f, -0.5f };
        ManipulatePosition(camera, &P0, 0.3f);
        ManipulatePosition(camera, &P2, 0.3f);

        DrawText(P0, "P0");
        DrawText(P2, "P2");

        // Brute force search 

        if(0)
        for (auto tri : deltaPolygonSoup.triangles)
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
        minimum_lbvh::NodeIndex currentNode = deltaPolygonSoup.builder.m_rootNode;

#if 0
        // reflection 1 level
        EventDescriptor eDescriptor;
        eDescriptor.set(0, Event::R);

        int numberOfNewton = 0;

        traverseAdmissibleNodes<1>(
            eDescriptor,
            1.0f,
            to(P0), to(P2),
            deltaPolygonSoup.builder.m_internals.data(),
            deltaPolygonSoup.internalsNormalBound.data(),
            deltaPolygonSoup.triangles.data(),
            deltaPolygonSoup.triangleAttribs.data(),
            deltaPolygonSoup.builder.m_rootNode,
            [&](AdmissibleTriangles<1> admissibleTriangles) {
                minimum_lbvh::Triangle tri = deltaPolygonSoup.triangles[admissibleTriangles.indices[0]];
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

                numberOfNewton++;

            });
        printf("numberOfNewton %d\n", numberOfNewton);
#endif

#if 1
        int numberOfNewton = 0;

        enum {
            K = 2
        };
        int index = 0;
        float eta = 1.3f;
        EventDescriptor eDescriptor;
        eDescriptor.set(0, Event::T);
        eDescriptor.set(1, Event::T);
        traverseAdmissibleNodes<K>(
            eDescriptor,
            eta,
            to(P0) /*light*/, to(P2),
            deltaPolygonSoup.builder.m_internals.data(),
            deltaPolygonSoup.internalsNormalBound.data(),
            deltaPolygonSoup.triangles.data(),
            deltaPolygonSoup.triangleAttribs.data(),
            deltaPolygonSoup.builder.m_rootNode,
            [&](AdmissibleTriangles<2> admissibleTriangles) {
                

                //bool debug = admissibleTriangles.indices[0] == 25 && admissibleTriangles.indices[1] == 91;
                //if (debug)
                //{
                //    printf("");
                //}


                //if (index++ != 12)
                //{
                //    return;
                //}
                //admissibleTriangles.indices[0] = 22;
                //admissibleTriangles.indices[1] = 15;
                minimum_lbvh::Triangle tri0 = deltaPolygonSoup.triangles[admissibleTriangles.indices[0]];
                minimum_lbvh::Triangle tri1 = deltaPolygonSoup.triangles[admissibleTriangles.indices[1]];

                if (debug_index == numberOfNewton)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        float3 v0 = tri0.vs[j];
                        float3 v1 = tri0.vs[(j + 1) % 3];
                        DrawLine(to(v0), to(v1), { 255, 0, 0 }, 4);
                    }
                    for (int j = 0; j < 3; ++j)
                    {
                        float3 v0 = tri1.vs[j];
                        float3 v1 = tri1.vs[(j + 1) % 3];
                        DrawLine(to(v0), to(v1), { 0, 255, 0 }, 2);
                    }
                }

                numberOfNewton++;

                //if (admissibleTriangles.indices[0] == 22 && admissibleTriangles.indices[1] == 15)
                ////if (admissibleTriangles.indices[0] == 15 && admissibleTriangles.indices[1] == 22)
                //{
                //    printf("");
                //}
                //minimum_lbvh::Triangle tri0 = deltaPolygonSoup.triangles[22];
                //minimum_lbvh::Triangle tri1 = deltaPolygonSoup.triangles[15];

                //float3 c0 = (tri0.vs[0] + tri0.vs[1] + tri0.vs[2]) / 3.0f;
                //float3 c1 = (tri1.vs[0] + tri1.vs[1] + tri1.vs[2]) / 3.0f;
                //DrawLine(to(c0), to(c1), { 255, 128, 0 }, 3);

                //DrawLine(P0, to(c0), { 255, 128, 0 }, 3);
                //DrawLine(to(c1), P2, { 255, 128, 0 }, 3);

                minimum_lbvh::Triangle tris[K];
                TriangleAttrib attribs[K];
                for (int k = 0; k < K; k++)
                {
                    int index = admissibleTriangles.indices[k];
                    tris[k] = deltaPolygonSoup.triangles[index];
                    attribs[k] = deltaPolygonSoup.triangleAttribs[index];
                }

                float parameters[4];
                bool converged = solveConstraints<K>(parameters, to(P0), to(P2), tris, attribs, eta, eDescriptor, 32, 1.0e-10f);

                if (converged)
                {
                    bool contributable = contributablePath<K>(
                        parameters, to(P0), to(P2), tris, attribs, eDescriptor,
                        polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode);

                    if (contributable)
                    {
                        float3 vertices[K + 2];
                        vertices[0] = to(P0);
                        vertices[K + 1] = to(P2);

                        for (int k = 0; k < K; k++)
                        {
                            float3 e0 = tris[k].vs[1] - tris[k].vs[0];
                            float3 e1 = tris[k].vs[2] - tris[k].vs[0];
                            float3 next = tris[k].vs[0] + e0 * parameters[k * 2 + 0] + e1 * parameters[k * 2 + 1];
                            vertices[k + 1] = next;
                        }

                        for (int k = 0; k < K + 1; k++)
                        {
                            DrawLine(to(vertices[k]), to(vertices[k + 1]), {255, 255, 0}, 3);
                        }

                        float3 ro = vertices[0];
                        float3 rd = vertices[1] - vertices[0];

                        for (int i = 0; i < 3; i++)
                        {
                            minimum_lbvh::Hit hit;
                            minimum_lbvh::intersect_stackfree(&hit, 
                                deltaPolygonSoup.builder.m_internals.data(), 
                                deltaPolygonSoup.triangles.data(), 
                                deltaPolygonSoup.builder.m_rootNode, ro, rd, 
                                minimum_lbvh::invRd(rd));

                            if (hit.t == MINIMUM_LBVH_FLT_MAX)
                            {
                                break;
                            }

                            TriangleAttrib attrib = deltaPolygonSoup.triangleAttribs[hit.triangleIndex];

                            float3 n =
                                attrib.shadingNormals[0] +
                                (attrib.shadingNormals[1] - attrib.shadingNormals[0]) * hit.uv.x +
                                (attrib.shadingNormals[2] - attrib.shadingNormals[0]) * hit.uv.y;

                            float thisEta = eta;
                            if (0.0f < dot(n, rd))
                            {
                                n = -n;
                                thisEta = 1.0f / thisEta;
                            }


                            float3 wi = -rd;
                            float3 wo;
                            if (refraction_norm_free(&wo, wi, n, thisEta))
                            {
                                DrawLine(to(ro), to(ro + rd * hit.t), { 255, 0, 0 }, 3);

                                ro = ro + rd * hit.t - normalize(n) * 0.000001f /* T */;
                                rd = wo;

                                DrawArrow(to(ro), to(ro + rd * 0.2f), 0.003f, {0, 255, 0});
                            }
                        }
                    }
                }
            });

            printf("numberOfNewton %d\n", numberOfNewton);
#endif


#endif 

#if 0
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

            enum {
                K = 2
            };
            minimum_lbvh::Triangle admissibleTriangles[K] = { tri1, tri0 };

            auto curved_dielectric = [](minimum_lbvh::Triangle tri, float curve) {
                float3 center = (tri.vs[0] + tri.vs[1] + tri.vs[2]) / 3.0f;

                float3 ng = minimum_lbvh::normalOf(tri);
                float3 n0 = normalize( ng + normalize(tri.vs[0] - center) * curve );
                float3 n1 = normalize( ng + normalize(tri.vs[1] - center) * curve );
                float3 n2 = normalize( ng + normalize(tri.vs[2] - center) * curve );

                return TriangleAttrib{ Material::Dielectric, { n0, n1, n2} };
            };

            TriangleAttrib admissibleAttribs[K] = {
                curved_dielectric(admissibleTriangles[0], 0.5f ),
                curved_dielectric(admissibleTriangles[1], 0.5f ),
            };

            for (int k = 0; k < K; k++)
            {
                
                for (int i = 0; i < 3; i++)
                {
                    float3 v = admissibleTriangles[k].vs[i];
                    float3 n = admissibleAttribs[k].shadingNormals[i];
                    DrawArrow(to(v), to(v + n), 0.01f, { 0, 255, 0 });
                }
            }

            EventDescriptor es;
            es.set(1, Event::T);
            es.set(0, Event::T);

            const int nParameters = K * 2;
            float parameters[nParameters];
            bool converged = solveConstraints<K>(parameters, to(P0), to(P2), admissibleTriangles, admissibleAttribs, 1.3f, es, 32, 1.0e-7f, [&](int iter, bool converged) {
                float3 vertices[K + 2];
                vertices[0] = to(P0);
                vertices[K + 1] = to(P2);

                float3 shadingNormals[K];

                for (int k = 0; k < K; k++)
                {
                    float param_u = parameters[k * 2 + 0];
                    float param_v = parameters[k * 2 + 1];

                    minimum_lbvh::Triangle tri = admissibleTriangles[k];

                    float3 e0 = tri.vs[1] - tri.vs[0];
                    float3 e1 = tri.vs[2] - tri.vs[0];
                    vertices[k + 1] = tri.vs[0] + e0 * param_u + e1 * param_v;

                    TriangleAttrib attrib = admissibleAttribs[k];
                    float3 ne0 = attrib.shadingNormals[1] - attrib.shadingNormals[0];
                    float3 ne1 = attrib.shadingNormals[2] - attrib.shadingNormals[0];
                    shadingNormals[k] = attrib.shadingNormals[0] + ne0 * param_u + ne1 * param_v;
                }

                for (int j = 0; j < K + 1; j++)
                {
                    glm::vec3 a = to(vertices[j]);
                    glm::vec3 b = to(vertices[j + 1]);
                    if (converged)
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

                // draw normal
                if (converged)
                {
                    for (int k = 0; k < K; k++)
                    {
                        float3 v = vertices[k + 1];
                        float3 n = shadingNormals[k];
                        DrawArrow(to(v), to(v + n * 0.2f), 0.005f, { 0, 255, 255 });
                    }
                }
            });
        }
#endif

        // Rendering
#if 1
        float3 light_intencity = { 1.0f, 1.0f, 1.0f };
        static glm::vec3 p_light = { 0, 1, 1 };
        ManipulatePosition(camera, &p_light, 0.3f);
        DrawText(p_light, "light");

        int stride = 2;
        Image2DRGBA8 image;
        image.allocate(GetScreenWidth() / stride, GetScreenHeight() / stride);

        CameraRayGenerator rayGenerator(GetCurrentViewMatrix(), GetCurrentProjMatrix(), image.width(), image.height());

        float eta = 1.3f;

        Stopwatch sw;

        // pre pass
        enum {
            MAX_PATH_LENGTH = 4,
            CACHE_STORAGE_COUNT = 1u << 18
        };

        struct TrianglePath 
        {
            uint32_t hashOfP;
            int tris[MAX_PATH_LENGTH];
        };
        std::vector<uint32_t> pathHashes(CACHE_STORAGE_COUNT);
        std::vector<TrianglePath> pathes(CACHE_STORAGE_COUNT);

        PCG rng;

        const float spacial_step = 0.025f;

        static int terminationCount = 32;

        // ray trace
#if 0
        enum {
            K = 1
        };
        EventDescriptor eDescriptor;
        eDescriptor.set(0, Event::R);
#else
        enum {
            K = 2
        };
        EventDescriptor eDescriptor;
        eDescriptor.set(0, Event::T);
        eDescriptor.set(1, Event::T);
#endif
        int totalPath = 0;
        for (int iTri = 0; iTri < polygonSoup.triangles.size(); iTri++)
        {
            minimum_lbvh::Triangle tri = polygonSoup.triangles[iTri];
            if (polygonSoup.triangleAttribs[iTri].material == Material::Diffuse)
            {
                continue;
            }

            int contiguousFails = 0;
            for (int j = 0; ; j++)
            {
                float2 params = {};
                sobol::shuffled_scrambled_sobol_2d(&params.x, &params.y, j, 123, 456, 789);
                params = square2triangle(params);

                float3 e0 = tri.vs[1] - tri.vs[0];
                float3 e1 = tri.vs[2] - tri.vs[0];
                float3 p = tri.vs[0] + e0 * params.x + e1 * params.y;

                float3 ro = to(p_light);
                float3 rd = p - to(p_light);

                bool inMedium = false;
                bool admissiblePath = false;
                int tris[K];
                int cacheTo = -1;
                float3 p_final;
                for (int d = 0; d < K + 1; d++)
                {
                    minimum_lbvh::Hit hit;
                    minimum_lbvh::intersect_stackfree(&hit, polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode, ro, rd, minimum_lbvh::invRd(rd));
                    if (hit.t == MINIMUM_LBVH_FLT_MAX)
                    {
                        break;
                    }
                    TriangleAttrib attrib = polygonSoup.triangleAttribs[hit.triangleIndex];
                    Material m = attrib.material;
                    float3 p_hit = ro + rd * hit.t;

                    float3 ns =
                        attrib.shadingNormals[0] +
                        (attrib.shadingNormals[1] - attrib.shadingNormals[0]) * hit.uv.x +
                        (attrib.shadingNormals[2] - attrib.shadingNormals[0]) * hit.uv.y;
                    float3 ng = dot(ns, hit.ng) < 0.0f ? -hit.ng : hit.ng; // aligned

                    if ( d == K )
                    {
                        if ( m == Material::Diffuse )
                        {
                            // store 
                            admissiblePath = true;
                            cacheTo = hit.triangleIndex;
                            p_final = p_hit;
                            // DrawPoint(to(p_hit), { 255, 0, 0 }, 2);
                        }
                        break;
                    }
                    if ( eDescriptor.get(d) == Event::R && (m == Material::Mirror || m == Material::Dielectric ))
                    {
                        tris[d] = hit.triangleIndex;

                        float3 wi = -rd;
                        float3 wo = reflection(wi, ns);

                        if (0.0f < dot(ng, wi) * dot(ng, wo)) // geometrically admissible
                        {
                            float3 ng_norm = normalize(ng);
                            ro = p_hit + (dot(wo, ng) < 0.0f ? -ng_norm : ng_norm) * 0.0001f;
                            rd = wo;
                            continue;
                        }
                    }
                    if (eDescriptor.get(d) == Event::T && m == Material::Dielectric )
                    {
                        tris[d] = hit.triangleIndex;

                        float3 wi = -rd;
                        float3 wo;

                        if (inMedium)
                        {
                            if (refraction_norm_free(&wo, wi, dot(ns, wi) < 0.0f ? -ns : ns, 1.0f / eta) == false)
                            {
                                break;
                            }
                        }
                        else
                        {
                            if (refraction_norm_free(&wo, wi, dot(ns, wi) < 0.0f ? -ns : ns, eta) == false)
                            {
                                break;
                            }
                        }
                        inMedium = !inMedium;

                        if (dot(ng, wi) * dot(ng, wo) < 0.0f) // geometrically admissible
                        {
                            float3 ng_norm = normalize(ng);
                            ro = p_hit + (dot(wo, ng) < 0.0f ? -ng_norm : ng_norm) * 0.0001f;
                            rd = wo;
                            continue;
                        }
                    }
                    break;
                }
                bool success = false;

                if (admissiblePath)
                {
                    uint32_t hashOfPath = 123;
                    for (int d = 0; d < K; d++)
                    {
                        hashOfPath = minimum_lbvh::hashPCG(hashOfPath + tris[d]);
                    }
                    hashOfPath |= 1u;

                    float3 indexf = (p_final / spacial_step);
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
                            if (pathHashes[index] == 0) // empty
                            {
                                pathHashes[index] = hashOfPath;
                                pathes[index].hashOfP = hashOfP;
                                for (int d = 0; d < K; d++)
                                {
                                    pathes[index].tris[d] = tris[d];
                                }
                                success = true;

                                totalPath++;
                                break;
                            }
                            else if (pathHashes[index] == hashOfPath)
                            {
                                break; // existing
                            }
                        }
                    }
                }

                if (success)
                {
                    contiguousFails = 0;
                }
                else
                {
                    contiguousFails++;
                }
                if (terminationCount < contiguousFails)
                {
                    break;
                }
            }
        }

        printf("occ %f\n", (float)totalPath / CACHE_STORAGE_COUNT);

        //for (int j = 0; j < image.height(); ++j){
        ParallelFor(image.height(), [&](int j) {
            for (int i = 0; i < image.width(); ++i)
            {
                bool debugPixel = i == 533 && j == 368;
                //if (!debugPixel)
                //{
                //    continue;
                //}

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

                if (polygonSoup.triangleAttribs[hit.triangleIndex].material != Material::Diffuse)
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

#if 0
                // reflection 1 level
                //EventDescriptor eDescriptor;
                //eDescriptor.set(0, Event::R);

                //traverseAdmissibleNodes<1>(
                //    eDescriptor,
                //    1.0f,
                //    p, to(p_light),
                //    deltaPolygonSoup.builder.m_internals.data(),
                //    deltaPolygonSoup.internalsNormalBound.data(),
                //    deltaPolygonSoup.triangles.data(),
                //    deltaPolygonSoup.triangleAttribs.data(),
                //    deltaPolygonSoup.builder.m_rootNode, 
                //    [&](AdmissibleTriangles<1> admissibleTriangles) {
                //        minimum_lbvh::Triangle tri = deltaPolygonSoup.triangles[admissibleTriangles.indices[0]];
                //        TriangleAttrib attrib = deltaPolygonSoup.triangleAttribs[admissibleTriangles.indices[0]];

                //        float parameters[2];
                //        bool converged = solveConstraints<1>(parameters, p, to(p_light), &tri, &attrib, eta, eDescriptor, 32, 1.0e-10f);

                //        if (converged )
                //        {
                //            bool contributable = contributablePath<1>(
                //                parameters, p, to(p_light), &tri, &attrib, eDescriptor,
                //                polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode);

                //            if (contributable)
                //            {
                //                minimum_lbvh::Triangle firstTri = tri;
                //                float3 e0 = firstTri.vs[1] - firstTri.vs[0];
                //                float3 e1 = firstTri.vs[2] - firstTri.vs[0];
                //                float3 firstHit = tri.vs[0] + e0 * parameters[0] + e1 * parameters[1];
                //                
                //                //float dAdwValue = 1.0f;
                //                float dAdwValue = dAdw(p, firstHit - p, to(p_light), &tri, &attrib, eDescriptor, 1, eta);
                //                L += reflectance * light_intencity / dAdwValue * fmaxf(dot(normalize(firstHit - p), n), 0.0f);
                //            }
                //        }
                //    });


                int numberOfNewton = 0;
                
                // linear probing
                uint32_t hashOfP = spacial_hash(p, spacial_step);
                uint32_t home = hashOfP % CACHE_STORAGE_COUNT;
                for (int offset = 0; offset < CACHE_STORAGE_COUNT; offset++)
                {
                    uint32_t index = (home + offset) % CACHE_STORAGE_COUNT;
                    if (pathHashes[index] == 0)
                    {
                        break; // no more cached
                    }

                    if (pathes[index].hashOfP != hashOfP)
                    {
                        continue;
                    }

                    numberOfNewton++;

                    minimum_lbvh::Triangle tris[K];
                    TriangleAttrib attribs[K];
                    for (int k = 0; k < K; k++)
                    {
                        int indexOfTri = pathes[index].tris[k];
                        tris[k] = polygonSoup.triangles[indexOfTri];
                        attribs[k] = polygonSoup.triangleAttribs[indexOfTri];
                    }

                    float parameters[K * 2];
                    bool converged = solveConstraints<1>(parameters, to(p_light), p, tris, attribs, eta, eDescriptor, 32, 1.0e-10f);

                    if (converged)
                    {
                        bool contributable = contributablePath<1>(
                            parameters, to(p_light), p, tris, attribs, eDescriptor,
                            polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode);

                        if (contributable)
                        {
                            float3 firstHit = getVertex(0, tris, parameters);

                            float dAdwValue = dAdw(to(p_light), firstHit - to(p_light), p, tris, attribs, eDescriptor, 1, eta);
                            L += reflectance * light_intencity / dAdwValue * fmaxf(dot(normalize(firstHit - p), n), 0.0f);
                        }
                    }
                }

#else
                int numberOfNewton = 0;

                // refraction 2 levels
                //EventDescriptor eDescriptor;
                //eDescriptor.set(0, Event::T);
                //eDescriptor.set(1, Event::T);

                //enum {
                //    K = 2
                //};
                //traverseAdmissibleNodes<K>(
                //    eDescriptor,
                //    eta,
                //    to(p_light), p,
                //    deltaPolygonSoup.builder.m_internals.data(),
                //    deltaPolygonSoup.internalsNormalBound.data(),
                //    deltaPolygonSoup.triangles.data(),
                //    deltaPolygonSoup.triangleAttribs.data(),
                //    deltaPolygonSoup.builder.m_rootNode,
                //    [&](AdmissibleTriangles<K> admissibleTriangles) {
                //        numberOfNewton++;

                //        minimum_lbvh::Triangle tris[K];
                //        TriangleAttrib attribs[K];
                //        for (int k = 0; k < K ; k++)
                //        {
                //            int index = admissibleTriangles.indices[k];
                //            tris[k] = deltaPolygonSoup.triangles[index];
                //            attribs[k] = deltaPolygonSoup.triangleAttribs[index];
                //        }

                //        float parameters[4];
                //        bool converged = solveConstraints<K>(parameters, to(p_light), p, tris, attribs, eta, eDescriptor, 32, 1.0e-10f);

                //        if (converged)
                //        {
                //            bool contributable = contributablePath<K>(
                //                parameters, to(p_light), p, tris, attribs, eDescriptor,
                //                polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode);

                //            if (contributable)
                //            {
                //                float dAdwValue = dAdw(to(p_light), getVertex(0, tris, parameters) - to(p_light), p, tris, attribs, eDescriptor, K, eta);
                //                L += reflectance * light_intencity / dAdwValue * fmaxf(dot(normalize(getVertex(K - 1, tris, parameters) - p), n), 0.0f);
                //            }
                //        }

                //});

                // linear probing
                uint32_t hashOfP = spacial_hash(p, spacial_step);
                uint32_t home = hashOfP % CACHE_STORAGE_COUNT;
                for (int offset = 0; offset < CACHE_STORAGE_COUNT; offset++)
                {
                    // CACHE_STORAGE_COUNT
                    uint32_t index = (home + offset) % CACHE_STORAGE_COUNT;
                    if (pathHashes[index] == 0)
                    {
                        break; // no more cached
                    }
                    if (pathes[index].hashOfP != hashOfP)
                    {
                        continue;
                    }
                    numberOfNewton++;

                    minimum_lbvh::Triangle tris[K];
                    TriangleAttrib attribs[K];
                    for (int k = 0; k < K; k++)
                    {
                        int indexOfTri = pathes[index].tris[k];
                        tris[k] = polygonSoup.triangles[indexOfTri];
                        attribs[k] = polygonSoup.triangleAttribs[indexOfTri];
                    }

                    float parameters[K * 2];
                    bool converged = solveConstraints<K>(parameters, to(p_light), p, tris, attribs, eta, eDescriptor, 32, 1.0e-10f);

                    if (converged)
                    {
                        bool contributable = contributablePath<K>(
                            parameters, to(p_light), p, tris, attribs, eDescriptor,
                            polygonSoup.builder.m_internals.data(), polygonSoup.triangles.data(), polygonSoup.builder.m_rootNode);

                        if (contributable)
                        {
                            float dAdwValue = dAdw(to(p_light), getVertex(0, tris, parameters) - to(p_light), p, tris, attribs, eDescriptor, K, eta);
                            L += reflectance * light_intencity / dAdwValue * fmaxf(dot(normalize(getVertex(K - 1, tris, parameters) - p), n), 0.0f);
                        }
                    }
                }
#endif

                // glm::vec3 color = viridis((float)numberOfNewton / 128);

                float3 color = clamp(L, 0.0f, 1.0f);
                image(i, j) = { 
                    255 * powf(color.x, 1.0f / 2.2f), 
                    255 * powf(color.y, 1.0f / 2.2f), 
                    255 * powf(color.z, 1.0f / 2.2f), 255};

            };
        }
        );

        printf("e %f\n", sw.elapsed());

        texture->upload(image);

        //pr::PrimBegin(pr::PrimitiveMode::Lines);

        //for (auto tri : polygonSoup.triangles)
        //{
        //    for (int j = 0; j < 3; ++j)
        //    {
        //        float3 v0 = tri.vs[j];
        //        float3 v1 = tri.vs[(j + 1) % 3];
        //        pr::PrimVertex(to(v0), { 255, 255, 255 });
        //        pr::PrimVertex(to(v1), { 255, 255, 255 });
        //    }
        //}

        //pr::PrimEnd();
#endif

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::Checkbox("g_bruteforce", &g_bruteforce);
        ImGui::InputInt("debug_index", &debug_index);
        ImGui::InputInt("terminationCount", &terminationCount);
        
        //ImGui::SliderFloat("param_a_init", &param_a_init, 0, 1);
        //ImGui::SliderFloat("param_b_init", &param_b_init, 0, 1);
        //if (ImGui::Button("restart"))
        //{
        //    param_a = 0.3f;
        //    param_b = 0.3f;
        //}

        if (ImGui::Button("save"))
        {
            image.saveAsPng("test.png");
        }

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
