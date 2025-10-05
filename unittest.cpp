#include "catch_amalgamated.hpp"
#include "interval.h"
#include "pk.h"

const float pi = 3.14159265358979323846f;

inline float random_float(uint32_t u)
{
    // take minimum s to satisfy [0.0, 1.0)
    enum {
        s = 8u,
        bad = s - 1
    };
    static_assert((float)(0xFFFFFFFF >> bad) / ((0xFFFFFFFF >> bad) + 1) == 1.0f, "");
    static_assert((float)(0xFFFFFFFF >> s) / ((0xFFFFFFFF >> s) + 1) == 0.99999994f, "");
    return (float)(u >> s) / ((0xFFFFFFFF >> s) + 1);
}

/*
 * PCG random number generator from
 * https://www.pcg-random.org/download.html#minimal-c-implementation
 */
struct PCG
{
    PCG(uint64_t seed, uint64_t sequence)
    {
        state = 0U;
        inc = (sequence << 1u) | 1u;

        uniform();
        state += seed;
        uniform();
    }

    uint32_t uniform()
    {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    float uniformf()
    {
        return random_float(uniform());
    }

    uint64_t state;  // RNG state.  All values are possible.
    uint64_t inc;    // Controls which RNG sequence(stream) is selected. Must *always* be odd.
};

TEST_CASE("square") {
    using namespace interval;

    PCG rng(103, 178);
    for (int i = 0; i < 1000; i++)
    {
        float x = lerp(-16.0f, 16.0f, rng.uniformf());
        intr x_interval(x, x + 1.0f);
        intr xx_interval = square(x_interval);

        float lower = FLT_MAX;
        float upper = -FLT_MAX;

        int NSample = 1000;
        for (int j = 0; j < NSample; j++)
        {
            float x_sample = lerp(x_interval.l, x_interval.u, (float)j / (NSample - 1) );
            float xx_sample = x_sample * x_sample;

            lower = std::min(lower, xx_sample);
            upper = std::max(upper, xx_sample);
        }

        REQUIRE(xx_interval.l <= lower); // inside check
        REQUIRE(upper <= xx_interval.u); // inside check

        float margin = 0.01f;
        REQUIRE(lower < xx_interval.l + margin ); // border check
        REQUIRE(xx_interval.u - margin < upper );  // border check
    }
}

TEST_CASE("normalize") {
    using namespace interval;

    PCG rng(103, 178);

    for (int i = 0; i < 1000; i++)
    {
        float size = lerp(0.05f, 0.4f, rng.uniformf());
        float x = lerp(-2.0f, 2.0f, rng.uniformf());
        float y = lerp(-2.0f, 2.0f, rng.uniformf());
        float z = lerp(-2.0f, 2.0f, rng.uniformf());
        intr3 p_interval = make_intr3({ x, x + size }, { y, y + size }, { z, z + size });
        intr3 p_interval_normalized = normalize(p_interval);

        for (int j = 0; j < 1000; j++)
        {
            float x_sample = lerp(p_interval.x.l, p_interval.x.u, rng.uniformf());
            float y_sample = lerp(p_interval.y.l, p_interval.y.u, rng.uniformf());
            float z_sample = lerp(p_interval.z.l, p_interval.z.u, rng.uniformf());

            float len = sqrtf(x_sample * x_sample + y_sample * y_sample + z_sample * z_sample);

            x_sample /= len;
            y_sample /= len;
            z_sample /= len;

            // conservative test
            REQUIRE(intersects(p_interval_normalized, make_intr3(x_sample, y_sample, z_sample), 0.00001f));
        }
    }
}

TEST_CASE("refract") {
    PCG rng(103, 178);

    for (int j = 0; j < 100; j++)
    {
        float eta = 1.0f + rng.uniformf();
        int N = 100;
        for (int i = 0; i < N; i++)
        {
            float theta = pi * 0.5f * i / N;
            float3 n = { 0, 0.1f + rng.uniformf() * 3.0f, 0};
            float3 wi = {
                cosf(theta) * n.x - sinf(theta) * n.y,
                sinf(theta) * n.x + cosf(theta) * n.y,
            };

            float R = fresnel_exact_norm_free(wi, n, eta);

            float3 wo = refraction_norm_free(wi, n, eta);

            float R_inv = fresnel_exact_norm_free(wo, -n, 1.0f / eta);

            REQUIRE( fabsf(R - R_inv) < 1.0e-3f );
        }
    }
}