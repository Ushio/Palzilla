#pragma once
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#define SOBOL_DEVICE __device__
#else
#define SOBOL_DEVICE
#include <stdint.h>
#endif

namespace sobol
{
    namespace details
    {
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
        using uint32_t = unsigned int;
#endif
        // Practical Hash-based Owen Scrambling
        // https://jcgt.org/published/0009/04/01/
        SOBOL_DEVICE inline uint32_t reverseBits(uint32_t v)
        {
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
            return __brev(v);
#else
            v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
            v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
            v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
            v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
            v = (v >> 16) | (v << 16);
            return v;
#endif
        }
        SOBOL_DEVICE inline uint32_t laine_karras_permutation(uint32_t x, uint32_t seed)
        {
            x += seed;
            x ^= x * 0x6c50b47cu;
            x ^= x * 0xb82f1e52u;
            x ^= x * 0xc7afe638u;
            x ^= x * 0x8d22f6e6u;
            return x;
        }
        SOBOL_DEVICE inline uint32_t nested_uniform_scramble(uint32_t x, uint32_t seed)
        {
            x = reverseBits(x);
            x = laine_karras_permutation(x, seed);
            x = reverseBits(x);
            return x;
        }

        // An Implementation Algorithm of 2D Sobol Sequence Fast, Elegant, and Compact
        // https://diglib.eg.org/items/57f2cdeb-69d9-434e-8cf8-37b63e7e69d9
        // Multiply Pascal Matrix
        SOBOL_DEVICE inline uint32_t P(uint32_t v /* take a reverse-bit index */)
        {
            v ^= v << 16;
            v ^= (v & 0x00FF00FF) << 8;
            v ^= (v & 0x0F0F0F0F) << 4;
            v ^= (v & 0x33333333) << 2;
            v ^= (v & 0x55555555) << 1;
            return v;
        }

        // [0.0, 1.0)
        SOBOL_DEVICE inline float random_float(uint32_t u)
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

        // Hash Functions for GPU Rendering
        // https://jcgt.org/published/0009/03/02/
        SOBOL_DEVICE inline uint32_t hashPCG(uint32_t v)
        {
            uint32_t state = v * 747796405 + 2891336453;
            uint32_t word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
            return (word >> 22) ^ word;
        }

        SOBOL_DEVICE inline void pcg3d(uint32_t* x, uint32_t* y, uint32_t* z) {
            uint32_t vx = *x;
            uint32_t vy = *y;
            uint32_t vz = *z;

            vx = vx * 1664525u + 1013904223u;
            vy = vy * 1664525u + 1013904223u;
            vz = vz * 1664525u + 1013904223u;

            vx += vy * vz;
            vy += vz * vx;
            vz += vx * vy;

            vx ^= vx >> 16u;
            vy ^= vy >> 16u;
            vz ^= vz >> 16u;

            vx += vy * vz;
            vy += vz * vx;
            vz += vx * vy;

            *x = vx;
            *y = vy;
            *z = vz;
        }
    }

    SOBOL_DEVICE inline void sobol_2d(float* x, float* y, uint32_t index)
    {
        using namespace details;

        uint32_t v /* van der corput sequence */ = reverseBits(index);
        *x = random_float(v);
        *y = random_float(P(v));
    }

    SOBOL_DEVICE inline void shuffled_scrambled_sobol_2d(float* x, float* y, uint32_t index, uint32_t p0, uint32_t p1, uint32_t p2)
    {
        using namespace details;

        pcg3d(&p0, &p1, &p2);

        uint32_t v /* van der corput sequence */ = laine_karras_permutation(reverseBits(index), p0);

        *x = random_float(nested_uniform_scramble(v, p1));
        *y = random_float(nested_uniform_scramble(P(v), p2));
    }
}