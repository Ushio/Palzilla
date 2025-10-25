#pragma once
#include "helper_math.h"

struct RayGenerator
{
    float3 m_origin = {0.0f, 0.0f, 0.0f};
    float3 m_right = {1.0f, 0.0f, 0.0f};
    float3 m_up = {0.0f, 1.0f, 0.0f};

    HELPER_MATH_DEVICE void lookat(float3 eye, float3 center, float3 up, float fovy, int imageWidth, int imageHeight)
    {
        float3 f(normalize(center - eye));
        float3 s(normalize(cross(f, up)));
        float3 u(cross(s, f));

        float tanThetaY = tan(fovy * 0.5f);
        float tanThetaX = tanThetaY / imageHeight * imageWidth;

        m_origin = eye;
        m_right = s * tanThetaX;
        m_up = u * tanThetaY;
    }

    HELPER_MATH_DEVICE void shoot(float3* ro, float3* rd, float u, float v)
    {
        float3 from = m_origin;
        float3 forward = normalize(cross(m_up, m_right));
        float3 to = m_origin + forward + lerp(-m_right, m_right, u) + lerp(m_up, -m_up, v);
        *ro = from;
        *rd = to - from;
    }
};
