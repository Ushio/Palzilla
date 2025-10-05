#include "helper_math.h"
#include <math.h>

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