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

    inline intr3 operator*(intr3 a, float s)
    {
        return intr3{
            a.x * s,
            a.y * s,
            a.z * s
        };
    }
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

    inline intr3 reflection(intr3 wi, intr3 n)
    {
        return n * dot(wi, n) * 2.0f - wi; 
    }

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
            {+INTERVAL_FLT_MAX, -INTERVAL_FLT_MAX},
            {+INTERVAL_FLT_MAX, -INTERVAL_FLT_MAX},
            {+INTERVAL_FLT_MAX, -INTERVAL_FLT_MAX},
        };

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
    }
}