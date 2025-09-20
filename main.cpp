#include "pr.hpp"
#include <iostream>
#include <memory>

#include "interval.h"
#include "helper_math.h"
#include "minimum_lbvh.h"

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
inline float3 reflection(float3 wi, float3 n)
{
    return n * dot(wi, n) * 2.0f / dot(n, n) - wi;
}

struct TriangleAttrib
{
    float3 shadingNormals[3];
};

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };

    double e = GetElapsedTime();


    SetDataDir(ExecutableDir());
    AbcArchive archive;
    std::string err;
    archive.open(GetDataPath("assets/scene.abc"), err);
    std::shared_ptr<FScene> scene = archive.readFlat(0, err);
    std::vector<minimum_lbvh::Triangle> triangles;
    minimum_lbvh::BVHCPUBuilder builder;
    std::vector<TriangleAttrib> triangleAttribs;

    scene->visitPolyMesh([&](std::shared_ptr<const FPolyMeshEntity> polymesh) {
        if (polymesh->visible() == false)
        {
            return;
        }
        ColumnView<int32_t> faceCounts(polymesh->faceCounts());
        ColumnView<int32_t> indices(polymesh->faceIndices());
        ColumnView<glm::vec3> positions(polymesh->positions());
        ColumnView<glm::vec3> normals(polymesh->normals());

        triangles.clear();
        triangleAttribs.clear();

        int indexBase = 0;
        for (int i = 0; i < faceCounts.count(); i++)
        {
            int nVerts = faceCounts[i];
            PR_ASSERT(nVerts == 3);
            minimum_lbvh::Triangle tri;
            TriangleAttrib attrib;
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

//        if (builder.empty())
//        {
//#if 1
//            Stopwatch sw;
//            builder.build(triangles.data(), triangles.size(), true /* isParallel */);
//            printf("build %f\n", sw.elapsed());
//
//            builder.validate();
//#else
//            Stopwatch sw;
//            builder.buildByEmbree(triangles.data(), triangles.size(), RTC_BUILD_QUALITY_LOW);
//            printf("embree build %f\n", sw.elapsed());
//#endif
//        }
        });

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

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
#if 1
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
        // test with mesh
        pr::PrimBegin(pr::PrimitiveMode::Lines);

        for (auto tri : triangles)
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

        for (auto tri : triangles)
        {
            interval::intr3 triangle_intr =
                interval::make_intr3(tri.vs[0].x, tri.vs[0].y, tri.vs[0].z) |
                interval::make_intr3(tri.vs[1].x, tri.vs[1].y, tri.vs[1].z) |
                interval::make_intr3(tri.vs[2].x, tri.vs[2].y, tri.vs[2].z);

            interval::intr3 wi_intr = interval::normalize(interval::make_intr3(P0.x, P0.y, P0.z) - triangle_intr);
            interval::intr3 wo_intr = interval::normalize(interval::make_intr3(P2.x, P2.y, P2.z) - triangle_intr);
            interval::intr3 H_intr = interval::normalize(wi_intr + wo_intr);

            float3 normal = normalize(cross(tri.vs[1] - tri.vs[0], tri.vs[2] - tri.vs[0]));

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

#endif 

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
