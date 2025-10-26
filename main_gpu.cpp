#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <stack>

#include "Orochi/Orochi.h"

#define ENABLE_GPU_BUILDER
#include "minimum_lbvh.h"

#include "typedbuffer.h"
#include "shader.h"

#include "camera.h"
#include "pk.h"

inline glm::vec3 to(float3 p)
{
    return { p.x, p.y, p.z };
}
inline float3 to(glm::vec3 p)
{
    return { p.x, p.y, p.z };
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

int main()
{
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 0;
    Initialize(config);
    SetDataDir(ExecutableDir());

    if (oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0))
    {
        printf("failed to init..\n");
        return 0;
    }

    int DEVICE_INDEX = 2;
    oroInit(0);
    oroDevice device;
    oroDeviceGet(&device, DEVICE_INDEX);
    oroCtx ctx;
    oroCtxCreate(&ctx, 0, device);
    oroCtxSetCurrent(ctx);

    oroDeviceProp props;
    oroGetDeviceProperties(&props, device);

    bool isNvidia = oroGetCurAPI(0) & ORO_API_CUDADRIVER;

    printf("Device: %s\n", props.name);
    printf("Cuda: %s\n", isNvidia ? "Yes" : "No");

    std::vector<std::string> options;
    options.push_back("-I");
    options.push_back(GetDataPath("../"));
    
    //options.push_back("-G");

    Shader shader(GetDataPath("../main_gpu.cu").c_str(), "main_gpu", options);

    TypedBuffer<uint32_t> pixels(TYPED_BUFFER_DEVICE);
    TypedBuffer<float4> accumulators(TYPED_BUFFER_DEVICE);

    // BVH
    minimum_lbvh::BVHGPUBuilder gpuBuilder(
        GetDataPath("../minimum_lbvh.cu").c_str(),
        GetDataPath("../").c_str()
    );
    tinyhiponesweep::OnesweepSort onesweep(device);

    std::vector<minimum_lbvh::Triangle> triangles;
    std::vector<TriangleAttrib> triangleAttribs;

    AbcArchive archive;
    std::string err;
    archive.open(GetDataPath("assets/scene.abc"), err);
    std::shared_ptr<FScene> scene = archive.readFlat(0, err);
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

            triangles.push_back(tri);
            triangleAttribs.push_back(attrib);
            indexBase += nVerts;
        }
    });

    TypedBuffer<minimum_lbvh::Triangle> trianglesDevice(TYPED_BUFFER_DEVICE);
    TypedBuffer<TriangleAttrib> triangleAttribsDevice(TYPED_BUFFER_DEVICE);
    trianglesDevice << triangles;
    triangleAttribsDevice << triangleAttribs;
    gpuBuilder.build(trianglesDevice.data(), trianglesDevice.size(), 0, onesweep, 0 /*stream*/);

    TypedBuffer<FirstDiffuse> firstDiffuses(TYPED_BUFFER_DEVICE);

    const float spacial_step = 0.025f;
    PathCache pathCache(TYPED_BUFFER_DEVICE);
    pathCache.init(spacial_step);

    TypedBuffer<float3> debugPoints(TYPED_BUFFER_DEVICE);
    TypedBuffer<int> debugPointCount(TYPED_BUFFER_DEVICE);
    debugPoints.allocate(1 << 22);
    debugPointCount.allocate(1);

    ITexture* texture = CreateTexture();

    Camera3D camera;
    camera.origin = { 2, 2, -2 };
    camera.lookat = { 0, 0, 0 };

    int iteration = 0;

    auto clearAccumulation = [&]() {
        oroMemsetD8(accumulators.data(), 0, accumulators.size() * sizeof(float4));
        iteration = 0;
    };

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            if (UpdateCameraBlenderLike(&camera))
            {
                clearAccumulation();
            }
        }

        //ClearBackground(0.1f, 0.1f, 0.1f, 1);
        ClearBackground(texture);

        BeginCamera(camera);

        PushGraphicState();

        {
            DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
            DrawXYZAxis(1.0f);
        }

        //pr::PrimBegin(pr::PrimitiveMode::Lines);

        //for (auto tri : triangles)
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

        // static glm::vec3 p_light = { 0, 2, 1 };
        static glm::vec3 p_light = { -0.580714, 0.861265, 1 };
        glm::vec3 prev_p_light = p_light;
        ManipulatePosition(camera, &p_light, 0.3f);
        DrawText(p_light, "light");
        if (prev_p_light != p_light)
        {
            clearAccumulation();
        }

        int imageWidth = GetScreenWidth();
        int imageHeight = GetScreenHeight();

        if (accumulators.size() != imageWidth * imageHeight)
        {
            pixels.allocate(imageWidth* imageHeight);
            accumulators.allocate(imageWidth * imageHeight);
            firstDiffuses.allocate(imageWidth* imageHeight);
            clearAccumulation();
        }

        RayGenerator rayGenerator;
        rayGenerator.lookat(to(camera.origin), to(camera.lookat), to(camera.up), camera.fovy, imageWidth, imageHeight);

        static float eta = 1.6f;
        static float minThroughput = 0.05f;

        DeviceStopwatch sw(0);

        sw.start();

        shader.launch("solvePrimary",
            ShaderArgument()
            .value(accumulators.data())
            .value(firstDiffuses.data())
            .value(int2{ imageWidth, imageHeight })
            .value(rayGenerator)
            .value(gpuBuilder.m_rootNode)
            .value(gpuBuilder.m_internals)
            .value(trianglesDevice.data())
            .value(triangleAttribsDevice.data())
            .value(to(p_light))
            .value(eta)
            .value(iteration),
            div_round_up64(imageWidth, 16), div_round_up64(imageHeight, 16), 1,
            16, 16, 1,
            0
        );

        sw.stop();
        printf("solvePrimary %f\n", sw.getElapsedMs());

        auto solveSpecular = [&](int K, EventDescriptor eDescriptor) {
            oroMemsetD32(debugPointCount.data(), 0, 1);

            sw.start();

            pathCache.clear();

            char photonTrace[128];
            char solveSpecular[128];
            sprintf(photonTrace, "photonTrace_K%d", K);
            sprintf(solveSpecular, "solveSpecular_K%d", K);

            shader.launch(photonTrace,
                ShaderArgument()
                .value(gpuBuilder.m_rootNode)
                .value(gpuBuilder.m_internals)
                .value(trianglesDevice.data())
                .value(triangleAttribsDevice.data())
                .value(to(p_light))
                .value(eDescriptor)
                .value(eta)
                .value(iteration)
                .ptr(&pathCache)
                .value(minThroughput)
                .value(debugPoints.data())
                .value(debugPointCount.data()),
                gpuBuilder.m_nTriangles, 1, 1,
                32, 1, 1,
                0
            );

            sw.stop();
            printf("%s %f\n", photonTrace, sw.getElapsedMs());

            printf(" occ %f\n", pathCache.occupancy());

            // debug view
            if (0)
            {
                int nPoints = 0;
                oroMemcpyDtoH(&nPoints, debugPointCount.data(), sizeof(int));
                std::vector<float3> points(nPoints);
                oroMemcpyDtoH(points.data(), debugPoints.data(), sizeof(float3) * nPoints);
                for (int i = 0; i < nPoints; i++)
                {
                    DrawPoint(to(points[i]), { 255, 0, 0 }, 2);
                }
            }

            sw.start();

            shader.launch(solveSpecular,
                ShaderArgument()
                .value(accumulators.data())
                .value(firstDiffuses.data())
                .value(int2{ imageWidth, imageHeight })
                .value(gpuBuilder.m_rootNode)
                .value(gpuBuilder.m_internals)
                .value(trianglesDevice.data())
                .value(triangleAttribsDevice.data())
                .value(to(p_light))
                .ptr(&pathCache)
                .value(eDescriptor)
                .value(eta)
                .value(iteration),
                div_round_up64(imageWidth, 16), div_round_up64(imageHeight, 16), 1,
                16, 16, 1,
                0
            );

            sw.stop();
            printf("%s %f\n", solveSpecular, sw.getElapsedMs());
        };

        //solveSpecular(1, { Event::R });
        solveSpecular(2, { Event::T, Event::T });
        //solveSpecular(3, { Event::T,Event::R, Event::T });
        //solveSpecular(4, { Event::T, Event::T, Event::T, Event::T });
        //solveSpecular(4, { Event::T, Event::R, Event::R, Event::T });

        //solveSpecular(1, { Event::R });
        //solveSpecular(2, { Event::R, Event::R });
        //solveSpecular(3, { Event::R, Event::R, Event::R });
        //solveSpecular(4, { Event::R, Event::R, Event::R, Event::R });

        shader.launch("pack",
            ShaderArgument()
            .value(pixels.data())
            .value(accumulators.data())
            .value(imageWidth* imageHeight),
            div_round_up64(imageWidth* imageHeight, 256), 1, 1,
            256, 1, 1,
            0
        );

        iteration++;

        static Image2DRGBA8 image;
        image.allocate(imageWidth, imageHeight);
        oroMemcpyDtoH(image.data(), pixels.data(), pixels.bytes());
        texture->upload(image);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 400, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::Text("spp : %d", iteration);
        ImGui::InputFloat("eta", &eta, 0.05f);
        if (ImGui::InputFloat("minThroughput", &minThroughput, 0.01f))
        {
            clearAccumulation();
        }
        
        ImGui::End();

        EndImGui();
    }
    return 0;
}