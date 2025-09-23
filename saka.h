#pragma once

namespace saka
{
    class dval
    {
    public:
        dval(): v(0.0f), g(0.0f) {}
        dval(float x) :v(x), g(0.0f) {}

        void requires_grad()
        {
            g = 1.0f;
        }

        float v;
        float g;
    };

    namespace details
    {
        template <class F, class dFdx>
        inline dval unary(dval x, F f, dFdx dfdx)
        {
            dval u;
            u.v = f(x.v);
            u.g = x.g * dfdx(x.v);
            return u;
        }

        template <class F, class dFdx, class dFdy>
        inline dval binary(dval x, dval y, F f, dFdx dfdx, dFdy dfdy)
        {
            dval u;
            u.v = f(x.v, y.v);
            u.g = x.g * dfdx(x.v, y.v) + y.g * dfdy(x.v, y.v);
            return u;
        }
    }
    inline dval operator+(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x + y; },
            [](float x, float y) { return 1.0f; }, // df/dx
            [](float x, float y) { return 1.0f; }  // df/dy
        );
    }
    inline dval operator-(dval x)
    {
        return details::unary(x,
            [](float x) { return -x; },
            [](float x) { return -1.0f; });
    }

    inline dval operator-(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x - y; },
            [](float x, float y) { return +1.0f; },
            [](float x, float y) { return -1.0f; });
    }
    inline dval operator*(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x * y; },
            [](float x, float y) { return y; }, // df/dx
            [](float x, float y) { return x; }  // df/dy
        );
    }
    inline dval operator/(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x / y; },
            [](float x, float y) { return 1.0f / y; },
            [](float x, float y) { return -x / (y * y); });
    }
    inline dval exp(dval x)
    {
        return details::unary(x, 
            [](float x) { return expf(x); }, 
            [](float x) { return expf(x); } // df/dx
        );
    }
    inline dval sqrt(dval x)
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

    inline dval3 make_dval3(dval x, dval y, dval z)
    {
        return { x, y, z };
    }

    template <class T>
    inline dval3 make_dval3(T v)
    {
        return { v.x, v.y, v.z };
    }

    inline dval3 operator+(dval3 a, dval3 b)
    {
        return {
            a.x + b.x,
            a.y + b.y,
            a.z + b.z
        };
    }

    inline dval3 operator-(dval3 a)
    {
        return {
            -a.x,
            -a.y,
            -a.z
        };
    }
    inline dval3 operator-(dval3 a, dval3 b)
    {
        return {
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        };
    }

    inline dval3 operator*(dval3 a, dval s)
    {
        return {
            a.x * s,
            a.y * s,
            a.z * s
        };
    }
    inline dval3 operator*(dval3 a, dval3 b)
    {
        return {
            a.x * b.x,
            a.y * b.y,
            a.z * b.z
        };
    }
    inline dval3 operator/(dval3 a, dval s)
    {
        return {
            a.x / s,
            a.y / s,
            a.z / s
        };
    }

    inline dval dot(dval3 a, dval3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    inline dval3 normalize(dval3 p)
    {
        auto len = sqrt(dot(p, p));
        return p / len;
    }
    inline dval3 cross(dval3 a, dval3 b)
    {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
}
