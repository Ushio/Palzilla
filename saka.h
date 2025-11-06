#pragma once

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define SAKA_DEVICE __device__
#else
#define SAKA_DEVICE
#endif

namespace saka
{
    class dval
    {
    public:
        SAKA_DEVICE dval(): v(0.0f), g(0.0f) {}
        SAKA_DEVICE dval(float x) : v(x), g(0.0f) {}
        SAKA_DEVICE dval(float x, bool requires_grad) :v(x), g(requires_grad) {}

        SAKA_DEVICE void requires_grad()
        {
            g = 1.0f;
        }

        float v;
        float g;
    };

    namespace details
    {
        template <class F, class dFdx>
        SAKA_DEVICE inline dval unary(dval x, F f, dFdx dfdx)
        {
            dval u;
            u.v = f(x.v);
            u.g = x.g * dfdx(x.v);
            return u;
        }

        template <class F, class dFdx, class dFdy>
        SAKA_DEVICE inline dval binary(dval x, dval y, F f, dFdx dfdx, dFdy dfdy)
        {
            dval u;
            u.v = f(x.v, y.v);
            u.g = x.g * dfdx(x.v, y.v) + y.g * dfdy(x.v, y.v);
            return u;
        }
    }
    SAKA_DEVICE inline dval operator+(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x + y; },
            [](float x, float y) { return 1.0f; }, // df/dx
            [](float x, float y) { return 1.0f; }  // df/dy
        );
    }
    SAKA_DEVICE inline dval operator-(dval x)
    {
        return details::unary(x,
            [](float x) { return -x; },
            [](float x) { return -1.0f; });
    }

    SAKA_DEVICE inline dval operator-(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x - y; },
            [](float x, float y) { return +1.0f; },
            [](float x, float y) { return -1.0f; });
    }
    SAKA_DEVICE inline dval operator*(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x * y; },
            [](float x, float y) { return y; }, // df/dx
            [](float x, float y) { return x; }  // df/dy
        );
    }
    SAKA_DEVICE inline dval operator/(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x / y; },
            [](float x, float y) { return 1.0f / y; },
            [](float x, float y) { return -x / (y * y); });
    }
    SAKA_DEVICE inline dval exp(dval x)
    {
        return details::unary(x, 
            [](float x) { return expf(x); }, 
            [](float x) { return expf(x); } // df/dx
        );
    }
    SAKA_DEVICE inline dval sqrt(dval x)
    {
        return details::unary(x, 
            [](float x) { return sqrtf(x); }, 
            [](float x) { return 0.5f / sqrtf(x); } // df/dx
        );
    }

    struct dval3
    {
        dval x;
        dval y;
        dval z;
    };

    SAKA_DEVICE inline dval3 make_dval3(dval x, dval y, dval z)
    {
        return { x, y, z };
    }

    template <class T>
    SAKA_DEVICE inline dval3 make_dval3(T v)
    {
        return { v.x, v.y, v.z };
    }

    SAKA_DEVICE inline dval3 operator+(dval3 a, dval3 b)
    {
        return {
            a.x + b.x,
            a.y + b.y,
            a.z + b.z
        };
    }

    SAKA_DEVICE inline dval3 operator-(dval3 a)
    {
        return {
            -a.x,
            -a.y,
            -a.z
        };
    }
    SAKA_DEVICE inline dval3 operator-(dval3 a, dval3 b)
    {
        return {
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        };
    }

    SAKA_DEVICE inline dval3 operator*(dval3 a, dval s)
    {
        return {
            a.x * s,
            a.y * s,
            a.z * s
        };
    }
    SAKA_DEVICE inline dval3 operator*(dval3 a, dval3 b)
    {
        return {
            a.x * b.x,
            a.y * b.y,
            a.z * b.z
        };
    }
    SAKA_DEVICE inline dval3 operator/(dval3 a, dval s)
    {
        return {
            a.x / s,
            a.y / s,
            a.z / s
        };
    }

    SAKA_DEVICE inline dval dot(dval3 a, dval3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    SAKA_DEVICE inline dval3 normalize(dval3 p)
    {
        auto len = sqrt(dot(p, p));
        return p / len;
    }
    SAKA_DEVICE inline dval3 cross(dval3 a, dval3 b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

    SAKA_DEVICE inline dval3 reflection(dval3 wi, dval3 n)
    {
        return n * dot(wi, n) * 2.0f / dot(n, n) - wi;
    }

    SAKA_DEVICE inline dval3 refraction_norm_free(dval3 wi, dval3 n, float eta /* = eta_t / eta_i */)
    {
        dval NoN = dot(n, n);
        dval WIoN = dot(wi, n);
        dval WoW = dot(wi, wi);
        dval k = NoN * WoW * (eta * eta - 1.0f) + WIoN * WIoN;
        if (k.v < 0.0f) // adhoc..
        {
            return { 0.0f, 0.0f, 0.0f };
        }
        return -wi * NoN + n * (WIoN - sqrt(k));
    }
}
