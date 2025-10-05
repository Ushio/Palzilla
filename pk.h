#include <math.h>
#include "helper_math.h"
#include "saka.h"
#include "minimum_lbvh.h"

enum class Material
{
    Diffuse,
    Mirror,
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