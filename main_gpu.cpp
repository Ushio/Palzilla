#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <stack>

#include "Orochi/Orochi.h"
#include "pk.h"

#if 1
// -- final render --
int main()
{
    using namespace pr;
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

    PKRenderer pkRenderer;
    pkRenderer.setup(device, GetDataPath("assets/scene.abc").c_str());
    pkRenderer.allocate(1920, 1080);

    for (int i = 0; i < pkRenderer.frameCount(); i++)
    {
        pkRenderer.loadFrame(i);
        pkRenderer.clear();

        for (int j = 0; j < 32; j++)
        {
            pkRenderer.step();
        }

        static Image2DRGBA8 image;
        pkRenderer.resolve(&image);

        char outputFile[128];
        sprintf(outputFile, "%03d.png", i);

        image.saveAsPng(outputFile);
    }
}
#else
// --interactive mode --
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

    ITexture* texture = CreateTexture();

    bool syncLight = true;
    int frameNumber = 50;

    PKRenderer pkRenderer;
    
    //pkRenderer.m_loadCamera = false;
    pkRenderer.m_camera.origin = { 1, 1, -1 };
    pkRenderer.m_camera.lookat = { 0, 0, 0 };

    pkRenderer.setup(device, GetDataPath("assets/scene.abc").c_str());
    pkRenderer.loadFrame(frameNumber);

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            if (UpdateCameraBlenderLike(&pkRenderer.m_camera))
            {
                pkRenderer.clear();
            }
        }

        //ClearBackground(0.1f, 0.1f, 0.1f, 1);
        ClearBackground(texture);

        BeginCamera(pkRenderer.m_camera);

        PushGraphicState();

        {
            DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
            DrawXYZAxis(1.0f);
        }

        //static glm::vec3 P2 = { 0.0109809, -0.1, -0.239754f };
        //ManipulatePosition(camera, &P2, 0.3f);
        //DrawText(P2, "P2");

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

        //static glm::vec3 P2 = { 0.0109809, -0.1, -0.239754f };
        //ManipulatePosition(camera, &P2, 0.1f);
        //DrawText(P2, "P2");

        glm::vec3 prev_p_light = pkRenderer.m_p_light;
        ManipulatePosition(pkRenderer.m_camera, &pkRenderer.m_p_light, 0.3f);
        DrawText(pkRenderer.m_p_light, "light");
        if (prev_p_light != pkRenderer.m_p_light)
        {
            pkRenderer.clear();
        }

        int imageWidth = GetScreenWidth();
        int imageHeight = GetScreenHeight();

        pkRenderer.allocate(imageWidth, imageHeight);
        pkRenderer.step();

        static Image2DRGBA8 image;
        pkRenderer.resolve(&image);
        texture->upload(image);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 400, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::Text("spp : %d", pkRenderer.m_iteration);
        if (ImGui::InputFloat("minThroughput", &pkRenderer.m_minThroughput, 0.01f))
        {
            pkRenderer.clear();
        }
        if (ImGui::SliderFloat("lightIntencity", &pkRenderer.m_lightIntencity, 0, 10))
        {
            pkRenderer.clear();
        }

        if (ImGui::InputInt("frameNumber", &frameNumber))
        {
            pkRenderer.loadFrame(frameNumber);
            pkRenderer.clear();
        }

        ImGui::Checkbox("loadCamera", &pkRenderer.m_loadCamera);
        
        ImGui::End();

        EndImGui();
    }
    return 0;
}

#endif