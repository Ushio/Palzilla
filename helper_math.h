/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define HELPER_MATH_LBVH_KERNELCC
#endif

#if defined(HELPER_MATH_LBVH_KERNELCC)
#define HELPER_MATH_HOST __host__
#define HELPER_MATH_DEVICE __device__

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;

#else
#include <inttypes.h>
#define HELPER_MATH_HOST
#define HELPER_MATH_DEVICE

struct alignas(8) int2
{
    int x, y;
};
struct int3
{
    int x, y, z;
};
struct alignas(16) int4
{
    int x, y, z, w;
};
struct alignas(8) uint2
{
    unsigned int x, y;
};
struct uint3
{
    unsigned int x, y, z;
};
struct alignas(16) uint4
{
    unsigned int x, y, z, w;
};

struct alignas(8) float2
{
    float x, y;
};
struct float3
{
    float x, y, z;
};
struct alignas(16) float4
{
    float x, y, z, w;
};

#define DECL_MAKE_234( structType, elementType )\
inline structType##2 make_##structType##2(elementType x, elementType y)\
{\
    return { x, y };\
}\
inline structType##3 make_##structType##3(elementType x, elementType y, elementType z)\
{\
    return { x, y, z };\
}\
inline structType##4 make_##structType##4(elementType x, elementType y, elementType z, elementType w)\
{\
    return { x, y, z, w };\
}

DECL_MAKE_234(float, float)
DECL_MAKE_234(int, int)
DECL_MAKE_234(uint, unsigned int)

#endif

//#include "cuda_runtime.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// bitwise
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 operator>>(uint2 a, uint32_t s)
{
    return { a.x >> s, a.y >> s };
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator^=(uint2& a, uint2 b)
{
    a.x ^= b.x;
    a.y ^= b.y;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 operator>>(uint3 a, uint32_t s)
{
    return { a.x >> s, a.y >> s, a.z >> s };
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator^=(uint3& a, uint3 b)
{
    a.x ^= b.x;
    a.y ^= b.y;
    a.z ^= b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 operator>>(uint4 a, uint32_t s)
{
    return { a.x >> s, a.y >> s, a.z >> s, a.w >> s };
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE void operator^=(uint4& a, uint4 b)
{
    a.x ^= b.x;
    a.y ^= b.y;
    a.z ^= b.z;
    a.w ^= b.w;
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  HELPER_MATH_HOST HELPER_MATH_DEVICE float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  HELPER_MATH_HOST HELPER_MATH_DEVICE float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_DEVICE HELPER_MATH_HOST float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_DEVICE HELPER_MATH_HOST float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline HELPER_MATH_DEVICE HELPER_MATH_HOST float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline HELPER_MATH_DEVICE HELPER_MATH_HOST int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline HELPER_MATH_DEVICE HELPER_MATH_HOST uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float length(float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float fracf(float v)
{
    return v - floorf(v);
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float2 fabs(float2 v)
{
    return make_float2(fabs(v.x), fabs(v.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline HELPER_MATH_HOST HELPER_MATH_DEVICE int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline HELPER_MATH_HOST HELPER_MATH_DEVICE int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_HOST HELPER_MATH_DEVICE float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline HELPER_MATH_DEVICE HELPER_MATH_HOST float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline HELPER_MATH_DEVICE HELPER_MATH_HOST float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

#endif
