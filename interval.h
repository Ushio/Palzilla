#pragma once

#define INTERVAL_FLT_MAX 3.402823466e+38F
namespace interval
{
    inline float ss_min(float a, float b)
    {
        return a < b ? a : b;
    }
    inline float ss_max(float a, float b)
    {
        return a > b ? a : b;
    }
    inline float ss_clamp(float x, float l, float u)
    {
        return ss_min(ss_max(x, l), u);
    }

    inline float ss_min_element(float a) {
        return a;
    }
    template <typename... Ts>
    inline float ss_min_element(float a, Ts... args) {
        return ss_min(a, ss_min_element(args...));
    }
    inline float ss_max_element(float a) {
        return a;
    }
    template <typename... Ts>
    inline float ss_max_element(float a, Ts... args) {
        return ss_max(a, ss_max_element(args...));
    }

    struct intr
    {
        intr() {}
        intr(float lower, float upper) :l(lower), u(upper)
        {
        }
        intr(float val) :l(val), u(val)
        {
        }
        float l;
        float u;
    };
    inline intr operator-(intr a)
    {
        return { -a.u, -a.l };
    }
    inline intr operator+(intr a, intr b)
    {
        return { a.l + b.l, a.u + b.u };
    }
    inline intr operator-(intr a, intr b)
    {
        return { a.l - b.u, a.u - b.l };
    }
    inline intr operator*(intr a, intr b)
    {
        float l = ss_min_element(
            a.l * b.l,
            a.l * b.u,
            a.u * b.l,
            a.u * b.u
        );
        float u = ss_max_element(
            a.l * b.l,
            a.l * b.u,
            a.u * b.l,
            a.u * b.u
        );
        return { l, u };
    }
    inline intr operator/(intr a, intr b)
    {
        return a * intr(
            1.0f / b.l,
            1.0f / b.u
        );
    }

    inline intr operator|(intr a, intr b)
    {
        return {
            ss_min(a.l, b.l),
            ss_max(a.u, b.u)
        };
    }

    inline intr sqrt(intr x)
    {
        return {
            sqrtf(x.l),
            sqrtf(x.u)
        };
    }
    inline intr square(intr x)
    {
        float l2 = x.l * x.l;
        float u2 = x.u * x.u;

        float larger = ss_max(l2, u2);
        if (x.l <= 0.0f && 0.0f <= x.u) // 0 included
        {
            return {
                0.0f,
                larger
            };
        }
        return {
            ss_min(l2, u2),
            larger
        };
    }

    struct intr3
    {
        intr x;
        intr y;
        intr z;
    };

    inline intr3 make_intr3(intr x, intr y, intr z)
    {
        return { x, y, z };
    }
    template <class T>
    inline intr3 make_intr3(T v)
    {
        return { v.x, v.y, v.z };
    }

    inline intr3 operator-(intr3 a)
    {
        return {
            - a.x,
            - a.y,
            - a.z
        };
    }


    inline intr3 operator|(intr3 a, intr3 b)
    {
        return intr3{
            a.x | b.x,
            a.y | b.y,
            a.z | b.z
        };
    }

    inline intr3 operator+(intr3 a, intr3 b)
    {
        return {
            a.x + b.x,
            a.y + b.y,
            a.z + b.z
        };
    }
    inline intr3 operator-(intr3 a, intr3 b)
    {
        return {
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        };
    }

    //inline intr3 operator*(intr3 a, float s)
    //{
    //    return intr3{
    //        a.x * s,
    //        a.y * s,
    //        a.z * s
    //    };
    //}
    inline intr3 operator*(intr3 a, intr s)
    {
        return intr3{
            a.x * s,
            a.y * s,
            a.z * s
        };
    }
    inline intr3 operator*(intr3 a, intr3 b)
    {
        return intr3{
            a.x * b.x,
            a.y * b.y,
            a.z * b.z
        };
    }
    inline intr3 operator/(intr3 a, intr s)
    {
        return intr3 {
            a.x / s,
            a.y / s,
            a.z / s
        };
    }

    inline intr dot(intr3 a, intr3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline intr3 relax(intr3 a, float s)
    {
        return make_intr3(
            { a.x.l - s, a.x.u + s },
            { a.y.l - s, a.y.u + s },
            { a.z.l - s, a.z.u + s }
        );
    }

    //inline intr3 reflection_naive(intr3 wi, intr3 n)
    //{
    //    return n * dot(wi, n) * 2.0f / dot(n, n) - wi;
    //}

    inline intr lengthSquared(intr3 v)
    {
        intr nxx = square(v.x);
        intr nyy = square(v.y);
        intr nzz = square(v.z);
        return nxx + nyy + nzz;
    }

    inline intr3 reflection(intr3 wi, intr3 n)
    {
        float eps = 1.0e-10f;
        intr nxx = square(n.x);
        intr nyy = square(n.y);
        intr nzz = square(n.z);
        intr nDotn = nxx + nyy + nzz;

        intr s = {
            2.0f / ss_max(nDotn.u, eps),
            2.0f / ss_max(nDotn.l, eps)
        };

        // reflection as a matrix
        intr m11 = s * nxx - intr(1.0f), m12 = s * n.x * n.y       , m13 = s * n.x * n.z;
        intr m21 = m12                 , m22 = s * nyy - intr(1.0f), m23 = s * n.y * n.z;
        intr m31 = m13                 , m32 = m23                 , m33 = s * nzz - intr(1.0f);

        return {
            m11 * wi.x + m12 * wi.y + m13 * wi.z,
            m21 * wi.x + m22 * wi.y + m23 * wi.z,
            m31 * wi.x + m32 * wi.y + m33 * wi.z
        };
    }

    inline intr3 refraction_normal(intr3 wi, intr3 wo, float eta /* = eta_t / eta_i */)
    {
        interval::intr lenWi = sqrt(interval::lengthSquared(wi));
        interval::intr lenWo = sqrt(interval::lengthSquared(wo));
        return -(wo * lenWi * eta + wi * lenWo);
    }

    inline intr3 refraction_normal_tight(intr3 wi, intr3 wo, float eta /* = eta_t / eta_i */)
    {
        interval::intr lenWi = sqrt(interval::lengthSquared(wi));
        interval::intr lenWo = sqrt(interval::lengthSquared(wo));

        interval::intr3 ht = -(wo * lenWi * eta + wi * lenWo);

        interval::intr lenWi_lenWo = lenWi * lenWo;

        // d ht.x / d wi.z
        // d ht.y / d wi.z
        // d ht.z / d wi.z
        interval::intr dhx_dwix = -lenWi_lenWo - wi.x * wo.x * eta;
        interval::intr dhy_dwiy = -lenWi_lenWo - wi.y * wo.y * eta;
        interval::intr dhz_dwiz = -lenWi_lenWo - wi.z * wo.z * eta;

        // d ht.x / d wo.x
        // d ht.y / d wo.y
        // d ht.z / d wo.z
        interval::intr dhx_dwox = -eta * lenWi_lenWo - wi.x * wo.x;
        interval::intr dhy_dwoy = -eta * lenWi_lenWo - wi.y * wo.y;
        interval::intr dhz_dwoz = -eta * lenWi_lenWo - wi.z * wo.z;


        // x bound lower
        {
            intr3 wi_bounded = wi;
            intr3 wo_bounded = wo;

            if (dhx_dwox.u < 0.0f) // negative monotonic
            {
                wo_bounded.x = wo.x.u;
            }
            if (dhx_dwix.u < 0.0f) // negative monotonic
            {
                wi_bounded.x = wi.x.u;
            }
            ht.x.l = refraction_normal(wi_bounded, wo_bounded, eta).x.l;
        }
        {
            intr3 wi_bounded = wi;
            intr3 wo_bounded = wo;

            if (dhx_dwox.u < 0.0f) // negative monotonic
            {
                wo_bounded.x = wo.x.l;
            }
            if (dhx_dwix.u < 0.0f) // negative monotonic
            {
                wi_bounded.x = wi.x.l;
            }
            ht.x.u = refraction_normal(wi_bounded, wo_bounded, eta).x.u;
        }

        // y bound lower
        {
            intr3 wi_bounded = wi;
            intr3 wo_bounded = wo;

            if (dhy_dwoy.u < 0.0f) // negative monotonic
            {
                wo_bounded.y = wo.y.u;
            }
            if (dhy_dwiy.u < 0.0f) // negative monotonic
            {
                wi_bounded.y = wi.y.u;
            }
            ht.y.l = refraction_normal(wi_bounded, wo_bounded, eta).y.l;
        }
        {
            intr3 wi_bounded = wi;
            intr3 wo_bounded = wo;

            if (dhy_dwoy.u < 0.0f) // negative monotonic
            {
                wo_bounded.y = wo.y.l;
            }
            if (dhy_dwiy.u < 0.0f) // negative monotonic
            {
                wi_bounded.y = wi.y.l;
            }
            ht.y.u = refraction_normal(wi_bounded, wo_bounded, eta).y.u;
        }

        // z bound lower
        {
            intr3 wi_bounded = wi;
            intr3 wo_bounded = wo;

            if (dhz_dwoz.u < 0.0f) // negative monotonic
            {
                wo_bounded.z = wo.z.u;
            }
            if (dhz_dwiz.u < 0.0f) // negative monotonic
            {
                wi_bounded.z = wi.z.u;
            }
            ht.z.l = refraction_normal(wi_bounded, wo_bounded, eta).z.l;
        }
        {
            intr3 wi_bounded = wi;
            intr3 wo_bounded = wo;

            if (dhz_dwoz.u < 0.0f) // negative monotonic
            {
                wo_bounded.z = wo.z.l;
            }
            if (dhz_dwiz.u < 0.0f) // negative monotonic
            {
                wi_bounded.z = wi.z.l;
            }
            ht.z.u = refraction_normal(wi_bounded, wo_bounded, eta).z.u;
        }

        return ht;
    }
    //inline intr3 refraction_norm_free(intr3 wi, intr3 n, float eta /* = eta_t / eta_i */)
    //{
    //    intr NoN = lengthSquared(n);
    //    intr WIoN = dot(wi, n);
    //    intr WIoWI = lengthSquared(wi);

    //    intr alpha = WIoN;
    //    intr beta = NoN * WIoWI;

    //    float g_lower = alpha.l - sqrtf(beta.u * (eta * eta - 1.0f) + alpha.l * alpha.l);
    //    float g_upper = alpha.u - sqrtf(beta.l * (eta * eta - 1.0f) + alpha.u * alpha.u);
    //    return -wi * NoN + n * intr{ g_lower, g_upper };
    //}
    //inline intr3 refraction_norm_free(intr3 wi, intr3 n, float eta /* = eta_t / eta_i */)
    //{
    //    intr NoN = lengthSquared(n);
    //    intr WIoN = dot(wi, n);
    //    intr WIoWI = lengthSquared(wi);
    //    intr k = NoN * WIoWI * (eta * eta - 1.0f) + square(WIoN);
    //    return -wi * NoN + n * (WIoN - sqrt(k));
    //}
    inline bool refraction_norm_free(intr3 *wo, intr3 wi, intr3 n, float eta /* = eta_t / eta_i */)
    {
        intr NoN = lengthSquared(n);
        intr WIoN = dot(wi, n);

        if (WIoN.u < 0.0f)
        {
            return false;
        }
        WIoN.l = ss_max(WIoN.l, 0.0f);

        intr WIoWI = lengthSquared(wi);
        intr k = NoN * WIoWI * (eta * eta - 1.0f) + square(WIoN);
        if (k.u < 0.0f)
        {
            return false;
        }
        k.l = ss_max(k.l, 0.0f);
        *wo = -wi * NoN + n * (WIoN - sqrt(k));
        return true;
    }
    //inline bool refraction_norm_free(intr3* wo, intr3 wi, intr3 n, float eta /* = eta_t / eta_i */)
    //{
    //    intr NoN = lengthSquared(n);
    //    intr WIoN = dot(wi, n);
    //    if (WIoN.u < 0.0f)
    //    {
    //        return false;
    //    }
    //    WIoN.l = ss_max(WIoN.l, 0.0f);

    //    intr WIoWI = lengthSquared(wi);

    //    intr alpha = WIoN;
    //    intr beta = NoN * WIoWI;

    //    if (eta < 1.0f)
    //    {
    //        return refraction_norm_free_n(wo, wi, n, eta);
    //    }
    //    else
    //    {
    //        float k_upper = beta.u * (eta * eta - 1.0f) + alpha.l * alpha.l;
    //        float k_lower = beta.l * (eta * eta - 1.0f) + alpha.u * alpha.u;
    //        if (k_upper < 0.0f)
    //        {
    //            return false;
    //        }
    //        k_lower = ss_max(k_lower, 0.0f);

    //        float g_lower = alpha.l - sqrtf(k_upper);
    //        float g_upper = alpha.u - sqrtf(k_lower);
    //        *wo = -wi * NoN + n * intr{ g_lower, g_upper };
    //    }

    //    return true;
    //}

    //inline intr3 refraction_norm_free(intr3 wi, intr3 n, float eta /* = eta_t / eta_i */)
    //{
    //    intr NoN = lengthSquared(n);
    //    intr WIoN = dot(wi, n);
    //    intr WIoWI = lengthSquared(wi);
    //    intr k = NoN * WIoWI * (eta * eta - 1.0f) + square(WIoN);

    //    intr y = (1.0 / eta) * n.y / n.z * wi.x - wi.y;
    //    intr c = (WIoN - sqrt(k));
    //    return -wi * NoN + n * (WIoN - sqrt(k));
    //}

    //inline intr3 normalize_naive(intr3 p)
    //{
    //    interval::intr len = interval::sqrt(
    //        interval::square(p.x) +
    //        interval::square(p.y) +
    //        interval::square(p.z)
    //    );
    //    return p / len;
    //}

    // Tighter than naiive.
    inline intr3 normalize(intr3 p)
    {
        const float eps = 1.0e-10f;

        if (p.x.l * p.x.u <= 0.0f && p.y.l * p.y.u <= 0.0f && p.z.l * p.z.u <= 0.0f)
        {
            return make_intr3(
                { -1.0f, 1.0f },
                { -1.0f, 1.0f },
                { -1.0f, 1.0f }
            );
        }

        float vs[3][2] = {
            {p.x.l, p.x.u},
            {p.y.l, p.y.u},
            {p.z.l, p.z.u},
        };
        float bound[3][2] = {
            {+2.0f, -2.0f},
            {+2.0f, -2.0f},
            {+2.0f, -2.0f},
        };

        // vertices
        for (int iz = 0; iz < 2; iz++)
        for (int iy = 0; iy < 2; iy++)
        for (int ix = 0; ix < 2; ix++)
        {
            float x = vs[0][ix];
            float y = vs[1][iy];
            float z = vs[2][iz];
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float len = ss_max(sqrtf(xx + yy + zz), eps);
            float nx = x / len;
            float ny = y / len;
            float nz = z / len;
            bound[0][0] = ss_min(bound[0][0], nx);
            bound[1][0] = ss_min(bound[1][0], ny);
            bound[2][0] = ss_min(bound[2][0], nz);
            bound[0][1] = ss_max(bound[0][1], nx);
            bound[1][1] = ss_max(bound[1][1], ny);
            bound[2][1] = ss_max(bound[2][1], nz);
        }

        // The box is at an octant
        if (0.0f < vs[0][0] * vs[0][1] && 0.0f < vs[1][0] * vs[1][1] && 0.0f < vs[2][0] * vs[2][1])
        {
            return make_intr3(
                { bound[0][0], bound[0][1] },
                { bound[1][0], bound[1][1] },
                { bound[2][0], bound[2][1] }
            );
        }

        // comments are xy plane case
        for (int axis = 0; axis < 3; axis++)
        {
            int axis_b = (axis + 1) % 3;
            int axis_c = (axis + 2) % 3;
            bool xyVisible = 0.0f < vs[axis][0] * vs[axis][1];
            if (xyVisible)
            {
                float near_z = 0.0f < vs[axis][0] ? vs[axis][0] : vs[axis][1];
                float near_zz = near_z * near_z;
                bool zeroInRangeX = vs[axis_b][0] * vs[axis_b][1] <= 0.0f;
                bool zeroInRangeY = vs[axis_c][0] * vs[axis_c][1] <= 0.0f;

                // -x, +x
                if (zeroInRangeY)
                {
                    float lower_x = vs[axis_b][0];
                    float upper_x = vs[axis_b][1];
                    float lower_len = ss_max(sqrtf(near_zz + lower_x * lower_x), eps);
                    float upper_len = ss_max(sqrtf(near_zz + upper_x * upper_x), eps);

                    // x
                    bound[axis_b][0] = ss_min(bound[axis_b][0], lower_x / lower_len);
                    bound[axis_b][1] = ss_max(bound[axis_b][1], upper_x / upper_len);
                }
                // -y, +y
                if (zeroInRangeX)
                {
                    float lower_y = vs[axis_c][0];
                    float upper_y = vs[axis_c][1];
                    float lower_len = ss_max(sqrtf(near_zz + lower_y * lower_y), eps);
                    float upper_len = ss_max(sqrtf(near_zz + upper_y * upper_y), eps);

                    // x
                    bound[axis_c][0] = ss_min(bound[axis_c][0], lower_y / lower_len);
                    bound[axis_c][1] = ss_max(bound[axis_c][1], upper_y / upper_len);
                }
                if (zeroInRangeX && zeroInRangeY)
                {
                    if (0.0f <= vs[axis][1])
                    {
                        bound[axis][1] = +1.0f;
                    }
                    else
                    {
                        bound[axis][0] = -1.0f;
                    }
                }
            }
        }

#if 0
        for (int iz = 0; iz < 2; iz++)
        for (int iy = 0; iy < 2; iy++)
        for (int ix = 0; ix < 2; ix++)
        {
            float x = vs[0][ix];
            float y = vs[1][iy];
            float z = vs[2][iz];
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float len = ss_max( sqrtf(xx + yy + zz), eps );
            float nx = x / len;
            float ny = y / len;
            float nz = z / len;
            bound[0][0] = ss_min(bound[0][0], nx);
            bound[1][0] = ss_min(bound[1][0], ny);
            bound[2][0] = ss_min(bound[2][0], nz);
            bound[0][1] = ss_max(bound[0][1], nx);
            bound[1][1] = ss_max(bound[1][1], ny);
            bound[2][1] = ss_max(bound[2][1], nz);

            // check middle. the bound is always on zero point.
            for (int ix_upper = ix + 1; ix_upper < 2; ix_upper++)
            {
                float x_upper = vs[0][ix_upper];
                if (x_upper * x <= 0.0f) // 0 included
                {
                    float len = ss_max(sqrtf(yy + zz), eps);
                    float ny = y / len;
                    float nz = z / len;
                    bound[1][0] = ss_min(bound[1][0], ny);
                    bound[2][0] = ss_min(bound[2][0], nz);
                    bound[1][1] = ss_max(bound[1][1], ny);
                    bound[2][1] = ss_max(bound[2][1], nz);
                }
            }
            for (int iy_upper = iy + 1; iy_upper < 2; iy_upper++)
            {
                float y_upper = vs[1][iy_upper];
                if (y_upper * y <= 0.0f) // 0 included
                {
                    float len = ss_max(sqrtf(xx + zz), eps);
                    float nx = x / len;
                    float nz = z / len;
                    bound[0][0] = ss_min(bound[0][0], nx);
                    bound[2][0] = ss_min(bound[2][0], nz);
                    bound[0][1] = ss_max(bound[0][1], nx);
                    bound[2][1] = ss_max(bound[2][1], nz);
                }
            }
            for (int iz_upper = iz + 1; iz_upper < 2; iz_upper++)
            {
                float z_upper = vs[2][iz_upper];
                if (z_upper * z <= 0.0f) // 0 included
                {
                    float len = ss_max(sqrtf(xx + yy), eps);
                    float nx = x / len;
                    float ny = y / len;
                    bound[0][0] = ss_min(bound[0][0], nx);
                    bound[1][0] = ss_min(bound[1][0], ny);
                    bound[0][1] = ss_max(bound[0][1], nx);
                    bound[1][1] = ss_max(bound[1][1], ny);
                }
            }
        }

        // mid point on the surface
        for (int axis = 0; axis < 3; axis++)
        {
            int axis_b = (axis + 1) % 3;
            int axis_c = (axis + 2) % 3;

            if (vs[axis_b][0] * vs[axis_b][1] <= 0.0f && vs[axis_c][0] * vs[axis_c][1] <= 0.0f)
            {
                if (0.0f <= vs[axis][1])
                {
                    bound[axis][1] = +1.0f;
                }
                else
                {
                    bound[axis][0] = -1.0f;
                }
            }
        }
#endif

        return make_intr3(
            { bound[0][0], bound[0][1] },
            { bound[1][0], bound[1][1] },
            { bound[2][0], bound[2][1] }
        );
    }

    inline bool intersects(intr3 a, intr3 b, float eps)
    {
        if (a.x.u + eps < b.x.l || b.x.u + eps < a.x.l)
        {
            return false;
        }
        if (a.y.u + eps < b.y.l || b.y.u + eps < a.y.l)
        {
            return false;
        }
        if (a.z.u + eps < b.z.l || b.z.u + eps < a.z.l)
        {
            return false;
        }
        return true;
    }

    inline bool zeroIncluded(intr3 v)
    {
        return 
            v.x.l * v.x.u <= 0.0f &&
            v.y.l * v.y.u <= 0.0f &&
            v.z.l * v.z.u <= 0.0f;
    }

    inline intr3 cross(intr3 a, intr3 b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
}