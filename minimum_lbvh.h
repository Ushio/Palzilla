#pragma once

#include "helper_math.h"

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define MINIMUM_LBVH_KERNELCC
#endif

#if defined(MINIMUM_LBVH_KERNELCC)

#define MINIMUM_LBVH_ASSERT(ExpectTrue) ((void)0)
#define MINIMUM_LBVH_DEVICE __device__

#else
#include <stdint.h>
#include <intrin.h>
#include <vector>
#include <Windows.h>
#ifdef DrawText
#undef DrawText
#endif
#include <ppl.h>
#define MINIMUM_LBVH_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { abort(); }
#define MINIMUM_LBVH_DEVICE

#if defined( ENABLE_EMBREE_BUILDER )
#include <embree4/rtcore.h>
#endif

#if defined( ENABLE_GPU_BUILDER )
#include "Orochi/Orochi.h"
#include "tinyhiponesweep.h"
#endif

#endif

#define MINIMUM_LBVH_FLT_MAX 3.402823466e+38F
#define MORTON_MAX_VALUE_3D 0x1FFFFF

namespace minimum_lbvh
{
	enum BUILD_OPTION
	{
		BUILD_OPTION_USE_NORMAL = 1,
		BUILD_OPTION_CPU_PARALLEL = 2
	};

	MINIMUM_LBVH_DEVICE uint64_t div_round_up64(uint64_t val, uint64_t divisor) noexcept { return (val + divisor - 1) / divisor; }
	MINIMUM_LBVH_DEVICE uint64_t next_multiple64(uint64_t val, uint64_t divisor) noexcept { return div_round_up64(val, divisor) * divisor; }

	template <class T>
	MINIMUM_LBVH_DEVICE void swap(T* a, T* b)
	{
		T tmp = *a;
		*a = *b;
		*b = tmp;
	}

	template <class T>
	MINIMUM_LBVH_DEVICE inline T ss_max(T x, T y)
	{
		return (x < y) ? y : x;
	}
	template <class T>
	MINIMUM_LBVH_DEVICE inline T ss_min(T x, T y)
	{
		return (y < x) ? y : x;
	}
	MINIMUM_LBVH_DEVICE inline float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
	{
		return (value - inputMin) * ((outputMax - outputMin) / (inputMax - inputMin)) + outputMin;
	}
	MINIMUM_LBVH_DEVICE inline float compMin(float3 v)
	{
		return fminf(fminf(v.x, v.y), v.z);
	}
	MINIMUM_LBVH_DEVICE inline float compMax(float3 v)
	{
		return fmaxf(fmaxf(v.x, v.y), v.z);
	}

	MINIMUM_LBVH_DEVICE inline float2 slabs(float3 ro, float3 one_over_rd, float3 lower, float3 upper, float knownHitT)
	{
		float3 t0 = (lower - ro) * one_over_rd;
		float3 t1 = (upper - ro) * one_over_rd;

		float3 tmin = fminf(t0, t1);
		float3 tmax = fmaxf(t0, t1);
		float region_min = compMax(tmin);
		float region_max = compMin(tmax) * 1.00000024f; // Robust BVH Ray Traversal- revised

		region_min = fmaxf(region_min, 0.0f);
		region_max = fminf(region_max, knownHitT);

		return { region_min, region_max };
	}

	// Return the number of consecutive high-order zero bits in a 32-bit integer
	MINIMUM_LBVH_DEVICE inline int clz(uint32_t x)
	{
#if !defined( MINIMUM_LBVH_KERNELCC )
		return __lzcnt(x);
#else
		return __clz(x);
#endif
	}
	MINIMUM_LBVH_DEVICE inline int clz64(uint64_t x)
	{
#if !defined( MINIMUM_LBVH_KERNELCC )
		return _lzcnt_u64(x);
#else
		return __clzll(x);
#endif
	}

	// Find the position of the least significant bit set to 1
	MINIMUM_LBVH_DEVICE inline int ffs(uint32_t x) {
#if !defined( MINIMUM_LBVH_KERNELCC )
		if (x == 0)
		{
			return 0;
		}
		return _tzcnt_u32(x) + 1;
#else
		return __ffs(x);
#endif
	}
	MINIMUM_LBVH_DEVICE inline int ffs64(uint64_t x) {
#if !defined( MINIMUM_LBVH_KERNELCC )
		if (x == 0)
		{
			return 0;
		}
		return _tzcnt_u64(x) + 1;
#else
		return __ffsll(x);
#endif
	}

	MINIMUM_LBVH_DEVICE inline int delta(uint32_t a, uint32_t b)
	{
		return 32 - clz(a ^ b);
	}
	MINIMUM_LBVH_DEVICE inline int delta(uint64_t a, uint64_t b)
	{
		return 64 - clz64(a ^ b);
	}

	MINIMUM_LBVH_DEVICE inline uint64_t encodeMortonCode_Naive(uint32_t x, uint32_t y, uint32_t z)
	{
		uint64_t code = 0;
		for (uint64_t i = 0; i < 64 / 3; ++i)
		{
			code |=
				((uint64_t)(x & (1u << i)) << (2 * i + 0)) |
				((uint64_t)(y & (1u << i)) << (2 * i + 1)) |
				((uint64_t)(z & (1u << i)) << (2 * i + 2));
		}
		return code;
	}

	MINIMUM_LBVH_DEVICE inline uint32_t compact3(uint64_t m)
	{
		uint64_t x = m & 0x1249249249249249;
		x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
		x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
		x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
		x = (x ^ (x >> 16)) & 0x1f00000000ffff;
		x = (x ^ (x >> 32)) & 0x1fffff;
		return static_cast<uint32_t>(x);
	}

	MINIMUM_LBVH_DEVICE inline uint64_t splat3(uint32_t a)
	{
		uint64_t x = a & 0x1fffff;
		x = (x | x << 32) & 0x1f00000000ffff;
		x = (x | x << 16) & 0x1f0000ff0000ff;
		x = (x | x << 8) & 0x100f00f00f00f00f;
		x = (x | x << 4) & 0x10c30c30c30c30c3;
		x = (x | x << 2) & 0x1249249249249249;
		return x;
	}

	MINIMUM_LBVH_DEVICE inline void decodeMortonCode(uint64_t morton, uint32_t* x, uint32_t* y, uint32_t* z)
	{
		*x = compact3(morton);
		*y = compact3(morton >> 1);
		*z = compact3(morton >> 2);
	}
	MINIMUM_LBVH_DEVICE inline uint64_t encodeMortonCode(uint32_t x, uint32_t y, uint32_t z)
	{
		uint64_t answer = 0;
		answer |= splat3(x) | splat3(y) << 1 | splat3(z) << 2;
		return answer;
	}

	MINIMUM_LBVH_DEVICE inline uint32_t encode32Morton2D(uint16_t x, uint16_t y)
	{
		uint64_t res = x | (uint64_t(y) << 32);
		res = (res | (res << 8)) & 0x00ff00ff00ff00ff;
		res = (res | (res << 4)) & 0x0f0f0f0f0f0f0f0f;
		res = (res | (res << 2)) & 0x3333333333333333;
		res = (res | (res << 1)) & 0x5555555555555555;
		return uint32_t(res | (res >> 31));
	}

	struct Triangle
	{
		float3 vs[3];
	};

	MINIMUM_LBVH_DEVICE inline float3 normalOf(const Triangle& tri)
	{
		return normalize(cross(tri.vs[1] - tri.vs[0], tri.vs[2] - tri.vs[0]));
	}

	struct AABB
	{
		float3 lower;
		float3 upper;

		MINIMUM_LBVH_DEVICE static AABB empty()
		{
			AABB aabb; 
			aabb.setEmpty();
			return aabb;
		}

		MINIMUM_LBVH_DEVICE void setEmpty()
		{
			lower = make_float3(+MINIMUM_LBVH_FLT_MAX);
			upper = make_float3(-MINIMUM_LBVH_FLT_MAX);
		}
		MINIMUM_LBVH_DEVICE void extend(const float3& p)
		{
			lower = fminf(lower, p);
			upper = fmaxf(upper, p);
		}

		MINIMUM_LBVH_DEVICE void extend(const AABB& b)
		{
			lower = fminf(lower, b.lower);
			upper = fmaxf(upper, b.upper);
		}

		MINIMUM_LBVH_DEVICE uint64_t encodeMortonCode(float3 p) const
		{
			int3 coord = make_int3((p - lower) / (upper - lower) * (float)(MORTON_MAX_VALUE_3D + 1));
			coord = clamp(coord, 0, MORTON_MAX_VALUE_3D);
			return minimum_lbvh::encodeMortonCode(coord.x, coord.y, coord.z);
		}
	};

	struct NodeIndex
	{
		MINIMUM_LBVH_DEVICE NodeIndex() :m_index(0), m_isLeaf(0) {}
		MINIMUM_LBVH_DEVICE NodeIndex(uint32_t index, bool isLeaf) :m_index(index), m_isLeaf(isLeaf) {}
		uint32_t m_index : 31;
		uint32_t m_isLeaf : 1;

		MINIMUM_LBVH_DEVICE static NodeIndex invalid()
		{
			return NodeIndex(0x7FFFFFFF, false);
		}
	};

	MINIMUM_LBVH_DEVICE inline bool operator==(NodeIndex a, NodeIndex b)
	{
		return a.m_index == b.m_index && a.m_isLeaf == b.m_isLeaf;
	}
	MINIMUM_LBVH_DEVICE inline bool operator!=(NodeIndex a, NodeIndex b)
	{
		return !(a == b);
	}

	struct InternalNode
	{
		NodeIndex children[2];
		AABB aabbs[2];
		NodeIndex parent;
		uint32_t context; // wasteful but for simplicity. upper bound of the range after building
	};

	MINIMUM_LBVH_DEVICE inline void build_lbvh(
		NodeIndex* rootNode,
		InternalNode* internals,
		const Triangle* triangles,
		int nTriangles,
		const uint8_t* deltas,
		uint32_t i_leaf,
		uint32_t triangleIndex )
	{
		int nInternals = nTriangles - 1;
		int nDeltas = nInternals;
		uint32_t leaf_lower = i_leaf;
		uint32_t leaf_upper = i_leaf;

		NodeIndex node(triangleIndex, true);

		AABB aabb; aabb.setEmpty();
		if (triangles)
		{
			for (auto v : triangles[triangleIndex].vs)
			{
				aabb.extend(v);
			}
		}

		bool isRoot = true;
		while (leaf_upper - leaf_lower < nInternals )
		{
			// direction from bottom
			uint32_t deltaL = 0 < leaf_lower ? deltas[leaf_lower - 1] : 0xFFFFFFFF;
			uint32_t deltaR = leaf_upper < nDeltas ? deltas[leaf_upper] : 0xFFFFFFFF;
			int goLeft = deltaL < deltaR ? 1 : 0;

			int parent = goLeft ? (leaf_lower - 1) : leaf_upper;

			internals[parent].children[goLeft] = node;
			internals[parent].aabbs[goLeft] = aabb;
			if (!node.m_isLeaf)
			{
				internals[node.m_index].parent = NodeIndex(parent, false);
			}

			uint32_t index = goLeft ? leaf_upper : leaf_lower;

			// == memory barrier ==

#if defined(MINIMUM_LBVH_KERNELCC)
			__threadfence();
			index = atomicExch(&internals[parent].context, index);
			__threadfence();
#else
			index = InterlockedExchange( &internals[parent].context, index );
#endif
			// == memory barrier ==

			if (index == 0xFFFFFFFF)
			{
				isRoot = false;
				break;
			}

			leaf_lower = ss_min(leaf_lower, index);
			leaf_upper = ss_max(leaf_upper, index);

			internals[parent].context = leaf_upper < nInternals ? leaf_upper : 0xFFFFFFFF;

			node = NodeIndex(parent, false);

			AABB otherAABB = internals[parent].aabbs[goLeft ^ 0x1];
			aabb.extend(otherAABB);
		}

		if (isRoot)
		{
			*rootNode = node;
			internals[node.m_index].parent = NodeIndex::invalid();
		}
	}

	MINIMUM_LBVH_DEVICE inline bool intersectRayTriangle(float* tOut, float* uOut, float* vOut, float3* ngOut, float t_min, float t_max, float3 ro, float3 rd, float3 v0, float3 v1, float3 v2 )
	{
		float3 e0 = v1 - v0;
		float3 e1 = v2 - v1;
		float3 ng = cross(e0, e1);

		float t = dot(v0 - ro, ng) / dot(ng, rd);
		if (t_min <= t && t <= t_max)
		{
			float3 e2 = v0 - v2;

			// Use tetrahedron volumes in space of 'ro' as the origin. note constant scale will be ignored.
			//   P0 = v0 - ro, P1 = v1 - ro, P2 = v2 - ro
			//   u_vol * 6 = ( P0 x P2 ) . rd
			//   v_vol * 6 = ( P1 x P0 ) . rd
			//   w_vol * 6 = ( P2 x P1 ) . rd
			// The cross product is unstable when ro is far away.. 
			// So let's use '2 ( a x b ) = (a - b) x (a + b)'
			//   u_vol * 12 = ( ( P0 - P2 ) x ( P0 + P2 ) ) . rd = ( ( P0 - P2 ) x ( v0 + v2 - ro * 2 ) ) . rd
			//   v_vol * 12 = ( ( P1 - P0 ) x ( P1 + P0 ) ) . rd = ( ( P1 - P0 ) x ( v1 + v0 - ro * 2 ) ) . rd
			//   w_vol * 12 = ( ( P2 - P1 ) x ( P2 + P1 ) ) . rd = ( ( P2 - P1 ) x ( v2 + v1 - ro * 2 ) ) . rd
			// As u, v, w volume are consistent on the neighbor, it is edge watertight.
			// Reference: https://github.com/RenderKit/embree/blob/v4.4.0/kernels/geometry/triangle_intersector_pluecker.h#L79-L94
			float u_vol = dot(cross(e2, v0 + v2 - ro * 2.0f), rd);
			float v_vol = dot(cross(e0, v1 + v0 - ro * 2.0f), rd);
			float w_vol = dot(cross(e1, v2 + v1 - ro * 2.0f), rd);

			// +,- mixed then no hits
			if (fminf(fminf(u_vol, v_vol), w_vol) < 0.0f && 0.0f < fmaxf(fmaxf(u_vol, v_vol), w_vol))
			{
				return false;
			}

			float vol = u_vol + v_vol + w_vol;

			// Barycentric Coordinates
			// hit = w*v0 + u*v1 + v*v2
			//     = v0 + u*(v1 - v0) + v*(v2 - v0)
			// float bW = w_vol / vol;
			float bU = u_vol / vol;
			float bV = v_vol / vol;

			*tOut = t;
			*uOut = bU;
			*vOut = bV;
			*ngOut = ng;
			return true;
		}

		return false;
	}

	MINIMUM_LBVH_DEVICE inline void validate_lbvh( NodeIndex node, const InternalNode* internals, const uint8_t* deltas, int maxDelta )
	{
		if (node.m_isLeaf)
		{
			return;
		}

		auto delta = deltas[node.m_index];
		MINIMUM_LBVH_ASSERT(delta <= maxDelta);
		validate_lbvh(internals[node.m_index].children[0], internals, deltas, delta);
		validate_lbvh(internals[node.m_index].children[1], internals, deltas, delta);
	}
#if defined( ENABLE_EMBREE_BUILDER )
	struct EmbreeBVHContext
	{
		EmbreeBVHContext() {
			nodes = 0;
			nodeHead = 0;
		}
		InternalNode* nodes;
		std::atomic<int> nodeHead;
	};

	inline void* node2ptr(NodeIndex node)
	{
		uint32_t data;
		memcpy(&data, &node, sizeof(uint32_t));
		return (char*)0 + data;
	}
	inline NodeIndex ptr2node(void* ptr)
	{
		uint32_t data = (char*)ptr - (char*)0;
		NodeIndex node;
		memcpy(&node, &data, sizeof(uint32_t));
		return node;
	}

	static void* embrreeCreateNode(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numChildren == 2);

		EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
		int index = context->nodeHead++;

		NodeIndex node(index, false);
		return node2ptr(node);
	}
	static void embreeSetNodeChildren(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numChildren == 2);

		EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
		NodeIndex theParentIndex = ptr2node(nodePtr);
		InternalNode& node = context->nodes[theParentIndex.m_index];
		for (int i = 0; i < numChildren; i++)
		{
			NodeIndex childIndex = ptr2node(childPtr[i]);
			node.children[i] = childIndex;

			// add a link child to parent
			if (childIndex.m_isLeaf == 0)
			{
				context->nodes[childIndex.m_index].parent = theParentIndex;
			}
		}
	}
	static void embreeSetNodeBounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numChildren == 2);

		EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
		InternalNode& node = context->nodes[ptr2node(nodePtr).m_index];

		for (int i = 0; i < numChildren; i++)
		{
			node.aabbs[i].lower = make_float3(bounds[i]->lower_x, bounds[i]->lower_y, bounds[i]->lower_z);
			node.aabbs[i].upper = make_float3(bounds[i]->upper_x, bounds[i]->upper_y, bounds[i]->upper_z);
		}
	}
	static void* embreeCreateLeaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numPrims == 1);
		NodeIndex node(prims->primID, true /*is leaf*/);
		return node2ptr(node);
	}
#endif

	struct IndexedMorton
	{
		uint32_t morton;
		uint32_t index;
	};

#if !defined(MINIMUM_LBVH_KERNELCC)
	class BVHCPUBuilder
	{
	public:
		void build(const Triangle *triangles, int nTriangles, uint32_t buildOption )
		{
			m_internals.clear();
			m_internals.resize(nTriangles - 1);
			for (int i = 0; i < m_internals.size(); i++)
			{
				m_internals[i].context = 0xFFFFFFFF;
			}
			m_deltas.resize(nTriangles - 1);

			// Scene AABB
			AABB sceneAABB;
			sceneAABB.setEmpty();

			for (int i = 0 ; i < nTriangles ; i++)
			{
				for (int j = 0; j < 3; ++j)
				{
					sceneAABB.extend(triangles[i].vs[j]);
				}
			}

			std::vector<IndexedMorton> mortons(nTriangles);
			for (int i = 0; i < nTriangles; i++)
			{
				Triangle tri = triangles[i];
				float3 center = (tri.vs[0] + tri.vs[1] + tri.vs[2]) / 3.0f;

				if (buildOption & BUILD_OPTION_USE_NORMAL)
				{
					float3 ng = normalOf(tri);
					AABB unit = { {-1, -1, -1}, {+1, +1, +1} };
					uint32_t nMorton = (uint32_t)(unit.encodeMortonCode(ng) >> 31);
					uint32_t pMorton = (uint32_t)(sceneAABB.encodeMortonCode(center) >> 31);
					mortons[i].morton = (pMorton & 0xFFFFFFC0) | (nMorton >> 26);
				}
				else
				{
					mortons[i].morton = (uint32_t)( sceneAABB.encodeMortonCode(center) >> 31 ); // take higher 32bits out of 63bits
				}
				mortons[i].index = i;
			}
			std::sort(mortons.begin(), mortons.end(), [](IndexedMorton a, IndexedMorton b) { return a.morton < b.morton; });

			for (int i = 0; i < m_deltas.size(); i++)
			{
				auto mA = mortons[i].morton;
				auto mB = mortons[i+1].morton;
				m_deltas[i] = delta(mA, mB);
			}

			if (buildOption & BUILD_OPTION_CPU_PARALLEL)
			{
				concurrency::parallel_for(size_t(0), mortons.size(), [&](uint32_t i_leaf) {
					build_lbvh(
						&m_rootNode,
						m_internals.data(),
						triangles,
						nTriangles,
						m_deltas.data(),
						i_leaf,
						mortons[i_leaf].index
					);
				});
			}
			else
			{
				for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
				{
					build_lbvh(
						&m_rootNode,
						m_internals.data(),
						triangles,
						nTriangles,
						m_deltas.data(),
						i_leaf,
						mortons[i_leaf].index
					);
				}
			}
		}

#if defined( ENABLE_EMBREE_BUILDER )
		void buildByEmbree(const Triangle* triangles, int nTriangles, RTCBuildQuality buildQuality)
		{
			RTCDevice device = rtcNewDevice("");
			RTCBVH bvh = rtcNewBVH(device);

			rtcSetDeviceErrorFunction(device, [](void* userPtr, RTCError code, const char* str) {
				printf("Embree Error [%d] %s\n", code, str);
			}, 0);

			std::vector<RTCBuildPrimitive> primitives(nTriangles);
			for (int i = 0; i < nTriangles; i++)
			{
				AABB aabb; aabb.setEmpty();
				for (auto v : triangles[i].vs)
				{
					aabb.extend(v);
				}
				RTCBuildPrimitive prim = {};
				prim.lower_x = aabb.lower.x;
				prim.lower_y = aabb.lower.y;
				prim.lower_z = aabb.lower.z;
				prim.geomID = 0;
				prim.upper_x = aabb.upper.x;
				prim.upper_y = aabb.upper.y;
				prim.upper_z = aabb.upper.z;
				prim.primID = i;
				primitives[i] = prim;
			}

			// allocation
			m_internals.clear();
			m_internals.resize(nTriangles  - 1);

			EmbreeBVHContext context;
			context.nodes = m_internals.data();

			RTCBuildArguments arguments = rtcDefaultBuildArguments();
			arguments.maxDepth = 64;
			arguments.byteSize = sizeof(arguments);
			arguments.buildQuality = buildQuality;
			arguments.maxBranchingFactor = 2;
			arguments.bvh = bvh;
			arguments.primitives = primitives.data();
			arguments.primitiveCount = primitives.size();
			arguments.primitiveArrayCapacity = primitives.size();
			arguments.minLeafSize = 1;
			arguments.maxLeafSize = 1;
			arguments.createNode = embrreeCreateNode;
			arguments.setNodeChildren = embreeSetNodeChildren;
			arguments.setNodeBounds = embreeSetNodeBounds;
			arguments.createLeaf = embreeCreateLeaf;
			arguments.splitPrimitive = nullptr;
			arguments.userPtr = &context;
			void* bvh_root = rtcBuildBVH(&arguments);

			rtcReleaseBVH(bvh);
			rtcReleaseDevice(device);

			m_rootNode = ptr2node(bvh_root);
			m_internals[m_rootNode.m_index].parent = NodeIndex::invalid();
		}
#endif
		bool empty() const
		{
			return m_internals.empty();
		}
		void validate() const
		{
			validate_lbvh(m_rootNode, m_internals.data(), m_deltas.data(), 255);
		}

		NodeIndex m_rootNode;
		std::vector<InternalNode> m_internals;
		std::vector<uint8_t> m_deltas;
	};
	
#if defined( ENABLE_GPU_BUILDER )
	inline void loadAsVector(std::vector<char>* buffer, const char* fllePath)
	{
		FILE* fp = fopen(fllePath, "rb");
		if (fp == nullptr) { return; }

		fseek(fp, 0, SEEK_END);

		buffer->resize(ftell(fp));

		fseek(fp, 0, SEEK_SET);

		size_t s = fread(buffer->data(), 1, buffer->size(), fp);
		if (s != buffer->size())
		{
			buffer->clear();
			return;
		}
		fclose(fp);
		fp = nullptr;
	}

	class DeviceStopwatch
	{
	public:
		DeviceStopwatch(oroStream stream)
		{
			m_stream = stream;
			oroEventCreateWithFlags(&m_start, oroEventDefault);
			oroEventCreateWithFlags(&m_stop, oroEventDefault);
		}
		~DeviceStopwatch()
		{
			oroEventDestroy(m_start);
			oroEventDestroy(m_stop);
		}
		DeviceStopwatch(const DeviceStopwatch&) = delete;
		void operator=(const DeviceStopwatch&) = delete;

		void start() { oroEventRecord(m_start, m_stream); }
		void stop() { oroEventRecord(m_stop, m_stream); }

		float getElapsedMs() const
		{
			oroEventSynchronize(m_stop);
			float ms = 0;
			oroEventElapsedTime(&ms, m_start, m_stop);
			return ms;
		}
	private:
		oroStream m_stream;
		oroEvent m_start;
		oroEvent m_stop;
	};

	class BVHGPUBuilder
	{
	public:
		BVHGPUBuilder(const char* kernel, const char* includeDir )
		{
			// load shader source code
			std::vector<char> src;
			loadAsVector(&src, kernel);
			MINIMUM_LBVH_ASSERT(0 < src.size());
			src.push_back('\0');

			orortcProgram program = 0;
			orortcCreateProgram(&program, src.data(), "builder", 0, 0, 0);

			std::vector<const char*> optionChars = {
				"-I",
				includeDir
			};
			orortcResult compileResult = orortcCompileProgram(program, optionChars.size(), optionChars.data());

			// print compilation log
			size_t logSize = 0;
			orortcGetProgramLogSize(program, &logSize);
			if (1 < logSize)
			{
				std::vector<char> compileLog(logSize);
				orortcGetProgramLog(program, compileLog.data());
				compileLog.push_back('\0');
				printf("%s", compileLog.data());
			}

			// get compiled code
			size_t codeSize = 0;
			orortcGetCodeSize(program, &codeSize);

			std::vector<char> codec(codeSize);
			orortcGetCode(program, codec.data());

			orortcDestroyProgram(&program);

			oroModuleLoadData(&m_module, codec.data());

			oroModuleGetFunction(&m_getSceneAABB, m_module, "getSceneAABB");
			oroModuleGetFunction(&m_buildMortons, m_module, "buildMortons");
			oroModuleGetFunction(&m_computeDeltas, m_module, "computeDeltas");
			oroModuleGetFunction(&m_build, m_module, "build");
			
			oroMalloc((void**)&m_sceneAABB, sizeof(AABB));
			oroMalloc((void**)&m_rootNode, sizeof(NodeIndex));
		}
		~BVHGPUBuilder()
		{
			oroFree(m_sceneAABB);
		}
		BVHGPUBuilder(const BVHGPUBuilder&) = delete;
		void operator=(const BVHGPUBuilder&) = delete;

		void build(const Triangle* triangles, int nTriangles, uint32_t buildOption, tinyhiponesweep::OnesweepSort& sorter, oroStream stream)
		{
			DeviceStopwatch sw(stream);
			sw.start();

			m_nTriangles = nTriangles;

			IndexedMorton* indexedMortons;
			IndexedMorton* indexedMortonsTmp;
			oroMallocAsync((void**)&indexedMortons,    sizeof(IndexedMorton) * nTriangles, stream);
			oroMallocAsync((void**)&indexedMortonsTmp, sizeof(IndexedMorton) * nTriangles, stream);

			uint8_t* deltas;
			oroMallocAsync((void**)&deltas, nTriangles - 1, stream);

			if (m_internals)
			{
				oroFree(m_internals);
			}
			oroMallocAsync((void**)&m_internals, sizeof(InternalNode) * (nTriangles - 1), stream);

			static const AABB emptyAABB = AABB::empty();
			oroMemcpyHtoDAsync(m_sceneAABB, (void*)&emptyAABB, sizeof(AABB), stream);

			{
				const void* args[] = {
					&m_sceneAABB,
					&triangles,
					&nTriangles
				};
				oroModuleLaunchKernel(m_getSceneAABB,
					div_round_up64(nTriangles, 256), 1, 1,
					256, 1, 1,
					0 /*shared*/, stream, args, 0 /*extras*/);
			}

			{
				const void* args[] = {
					&indexedMortons,
					&triangles,
					&nTriangles,
					&m_sceneAABB,
					&buildOption
				};
				oroModuleLaunchKernel(m_buildMortons,
					div_round_up64(nTriangles, 256), 1, 1,
					256, 1, 1,
					0 /*shared*/, stream, args, 0 /*extras*/);
			}

			sorter.sort({ (uint64_t *)indexedMortons, 0 }, { (uint64_t*)indexedMortonsTmp, 0 }, nTriangles, 0, sizeof(uint32_t) * 8, stream);

			{
				const void* args[] = {
					&deltas,
					&indexedMortons,
					&nTriangles
				};
				oroModuleLaunchKernel(m_computeDeltas,
					div_round_up64(nTriangles - 1, 256), 1, 1,
					256, 1, 1,
					0 /*shared*/, stream, args, 0 /*extras*/);
			}

			oroMemsetD32Async(m_internals, 0xFFFFFFFF, sizeof(InternalNode) * (nTriangles - 1) / 4, stream);


			{
				const void* args[] = {
					&m_rootNode,
					&m_internals,
					&triangles,
					&nTriangles,
					&deltas,
					&indexedMortons
				};
				oroModuleLaunchKernel(m_build,
					div_round_up64(nTriangles - 1, 256), 1, 1,
					256, 1, 1,
					0 /*shared*/, stream, args, 0 /*extras*/);
			}

			oroFreeAsync(indexedMortons, stream);
			oroFreeAsync(indexedMortonsTmp, stream);
			oroFreeAsync(deltas, stream);

			sw.stop();
			printf("%f ms\n", sw.getElapsedMs());
		}
		bool empty() const
		{
			return m_nTriangles == 0;
		}
		oroModule m_module = 0;
		oroFunction m_getSceneAABB = 0;
		oroFunction m_buildMortons = 0;
		oroFunction m_computeDeltas = 0;
		oroFunction m_build = 0;
		AABB* m_sceneAABB = 0;
		InternalNode* m_internals = 0;
		NodeIndex* m_rootNode;
		int m_nTriangles = 0;
	};
#endif

#endif

	struct Hit
	{
		float t = MINIMUM_LBVH_FLT_MAX;
		float2 uv = {};
		float3 ng = {};
		uint32_t triangleIndex = 0xFFFFFFFF;
	};

	MINIMUM_LBVH_DEVICE inline float3 invRd(float3 rd)
	{
		return clamp(1.0f / rd, -MINIMUM_LBVH_FLT_MAX, MINIMUM_LBVH_FLT_MAX);
	}

	// https://jcgt.org/published/0009/03/02/
	MINIMUM_LBVH_DEVICE inline uint32_t hashPCG(uint32_t v)
	{
		uint32_t state = v * 747796405 + 2891336453;
		uint32_t word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
		return (word >> 22) ^ word;
	}

	// stackful traversal for reference
	MINIMUM_LBVH_DEVICE inline void intersect_stackfull(
		Hit* hit,
		const InternalNode* nodes,
		const Triangle* triangles,
		NodeIndex rootNode,
		float3 ro,
		float3 rd,
		float3 one_over_rd, 
		NodeIndex* stack)
	{
		int sp = 0;

		NodeIndex node = rootNode;
		while (node != NodeIndex::invalid())
		{
			if (node.m_isLeaf)
			{
				float t;
				float u, v;
				float3 ng;
				const Triangle& tri = triangles[node.m_index];
				if (intersectRayTriangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
				{
					hit->t = t;
					hit->uv = make_float2(u, v);
					hit->ng = ng;
					hit->triangleIndex = node.m_index;
				}

				node = sp == 0 ? NodeIndex::invalid() : stack[--sp];
				continue;
			}

			const AABB& L = nodes[node.m_index].aabbs[0];
			const AABB& R = nodes[node.m_index].aabbs[1];

			float2 rangeL = slabs(ro, one_over_rd, L.lower, L.upper, hit->t);
			float2 rangeR = slabs(ro, one_over_rd, R.lower, R.upper, hit->t);
			bool hitL = rangeL.x <= rangeL.y;
			bool hitR = rangeR.x <= rangeR.y;

			if (hitL && hitR)
			{
				NodeIndex near_node = nodes[node.m_index].children[0];
				NodeIndex far_node  = nodes[node.m_index].children[1];
				if (rangeR.x < rangeL.x)
				{
					swap(&near_node, &far_node);
				}
				node = near_node;
				stack[sp++] = far_node;
			}
			else if (hitL || hitR)
			{
				node = nodes[node.m_index].children[hitL ? 0 : 1];
			}
			else
			{
				node = sp == 0 ? NodeIndex::invalid() : stack[--sp];
			}
		}
	}

	enum RAY_QUERY
	{
		RAY_QUERY_CLOSEST,
		RAY_QUERY_ANY
	};

	MINIMUM_LBVH_DEVICE inline void intersect_stackfree(
		Hit* hit,
		const InternalNode* nodes,
		const Triangle* triangles,
		NodeIndex node,
		float3 ro,
		float3 rd,
		float3 one_over_rd,
		RAY_QUERY rayQuery = RAY_QUERY_CLOSEST)
	{
		NodeIndex curr_node = node;
		NodeIndex prev_node = NodeIndex::invalid();

		while (curr_node != NodeIndex::invalid())
		{
			if (curr_node.m_isLeaf)
			{
				float t;
				float u, v;
				float3 ng;
				const Triangle& tri = triangles[curr_node.m_index];
				if (intersectRayTriangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
				{
					hit->t = t;
					hit->uv = make_float2(u, v);
					hit->triangleIndex = curr_node.m_index;
					hit->ng = ng;

					if (rayQuery == RAY_QUERY_ANY)
					{
						break;
					}
				}
				swap(&curr_node, &prev_node);
				continue;
			}

			AABB L = nodes[curr_node.m_index].aabbs[0];
			AABB R = nodes[curr_node.m_index].aabbs[1];
			float2 rangeL = slabs(ro, one_over_rd, L.lower, L.upper, hit->t); // far object may be culled during backtracking, but then nHits is 1 or 0 thus just go parent as expected.
			float2 rangeR = slabs(ro, one_over_rd, R.lower, R.upper, hit->t);
			bool hitL = rangeL.x <= rangeL.y;
			bool hitR = rangeR.x <= rangeR.y;

			NodeIndex parent_node = nodes[curr_node.m_index].parent;
			NodeIndex near_node = nodes[curr_node.m_index].children[0];
			NodeIndex far_node = nodes[curr_node.m_index].children[1];

			int nHits = 0;
			if (hitL && hitR)
			{
				if (rangeR.x < rangeL.x)
				{
					swap(&near_node, &far_node);
				}
				nHits = 2;
			}
			else if (hitL || hitR)
			{
				nHits = 1;
				near_node = hitR ? far_node : near_node;
			}

			NodeIndex next_node;
			if (prev_node == parent_node)
			{
				next_node = 0 < nHits ? near_node : parent_node;
			}
			else if (prev_node == near_node)
			{
				next_node = nHits == 2 ? far_node : parent_node;
			}
			else
			{
				next_node = parent_node;
			}

			prev_node = curr_node;
			curr_node = next_node;
		}
	}

	MINIMUM_LBVH_DEVICE inline void intersect_escape_link(
		Hit* hit,
		const InternalNode* nodes,
		const Triangle* triangles,
		NodeIndex node,
		float3 ro,
		float3 rd,
		float3 one_over_rd,
		RAY_QUERY rayQuery = RAY_QUERY_CLOSEST)
	{
		NodeIndex curr_node = node;
		NodeIndex prev_node = NodeIndex::invalid();

		while (curr_node != NodeIndex::invalid())
		{
			if (curr_node.m_isLeaf)
			{
				float t;
				float u, v;
				float3 ng;
				const Triangle& tri = triangles[curr_node.m_index];
				if (intersectRayTriangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
				{
					hit->t = t;
					hit->uv = make_float2(u, v);
					hit->triangleIndex = curr_node.m_index;
					hit->ng = ng;

					if (rayQuery == RAY_QUERY_ANY)
					{
						break;
					}
				}

				if (nodes[prev_node.m_index].children[0] == curr_node) // just go next
				{
					swap(&curr_node, &prev_node);
					continue;
				}

				uint32_t indexOfInternal = nodes[prev_node.m_index].context;
				NodeIndex next_node = indexOfInternal != 0xFFFFFFFF ? NodeIndex(indexOfInternal, false) : NodeIndex::invalid();

				prev_node = curr_node;
				curr_node = next_node;
				continue;
			}

			NodeIndex parent_node = nodes[curr_node.m_index].parent;
			AABB aabb;
			NodeIndex hitChild;
			NodeIndex miss;
			bool decent = prev_node == parent_node;
			if (decent)
			{
				aabb = nodes[curr_node.m_index].aabbs[0];
				hitChild = nodes[curr_node.m_index].children[0];
				miss = curr_node;
			}
			else
			{
				aabb = nodes[curr_node.m_index].aabbs[1];
				hitChild = nodes[curr_node.m_index].children[1];

				uint32_t indexOfInternal = nodes[curr_node.m_index].context;
				miss = indexOfInternal != 0xFFFFFFFF ? NodeIndex(indexOfInternal, false) : NodeIndex::invalid();
			}
			float2 range = slabs(ro, one_over_rd, aabb.lower, aabb.upper, hit->t);

			NodeIndex next_node = range.x <= range.y ? hitChild : miss;

			prev_node = curr_node;
			curr_node = next_node;
		}
	}
}
