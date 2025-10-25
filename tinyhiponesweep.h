#pragma once

#include "Orochi/Orochi.h"

namespace tinyhiponesweep
{
	constexpr int N_RADIX = 8;
	constexpr int BIN_SIZE = 1 << N_RADIX;
	constexpr int RADIX_MASK = (1 << N_RADIX) - 1;

	static_assert(BIN_SIZE % 2 == 0, "");

	constexpr int WARP_SIZE = 32;

	constexpr int RADIX_SORT_BLOCK_SIZE = 4096;

	constexpr int ONESWEEP_COUNT_ITEM_PER_BLOCK = 2048;
	constexpr int ONESWEEP_COUNT_THREADS_PER_BLOCK = 256;
	constexpr int ONESWEEP_COUNT_ITEMS_PER_THREAD = ONESWEEP_COUNT_ITEM_PER_BLOCK / ONESWEEP_COUNT_THREADS_PER_BLOCK;

	constexpr int REORDER_NUMBER_OF_WARPS = 8;
	constexpr int REORDER_NUMBER_OF_THREADS_PER_BLOCK = WARP_SIZE * REORDER_NUMBER_OF_WARPS;
	constexpr int REORDER_NUMBER_OF_ITEM_PER_WARP = RADIX_SORT_BLOCK_SIZE / REORDER_NUMBER_OF_WARPS;
	constexpr int REORDER_NUMBER_OF_ITEM_PER_THREAD = REORDER_NUMBER_OF_ITEM_PER_WARP / 32;

	constexpr int LOOKBACK_TABLE_SIZE = 1024;
	constexpr int MAX_LOOK_BACK = 64;
	constexpr int TAIL_BITS = 5;
	constexpr auto TAIL_MASK = 0xFFFFFFFFu << TAIL_BITS;
	static_assert(MAX_LOOK_BACK < LOOKBACK_TABLE_SIZE, "");

const char* kernel = R"(
constexpr int N_RADIX = 8;
constexpr int BIN_SIZE = 1 << N_RADIX;
constexpr int RADIX_MASK = ( 1 << N_RADIX ) - 1;

static_assert( BIN_SIZE % 2 == 0, "" );

constexpr int WARP_SIZE = 32;

constexpr int RADIX_SORT_BLOCK_SIZE = 4096;

constexpr int ONESWEEP_COUNT_ITEM_PER_BLOCK = 2048;
constexpr int ONESWEEP_COUNT_THREADS_PER_BLOCK = 256;
constexpr int ONESWEEP_COUNT_ITEMS_PER_THREAD = ONESWEEP_COUNT_ITEM_PER_BLOCK / ONESWEEP_COUNT_THREADS_PER_BLOCK;

constexpr int REORDER_NUMBER_OF_WARPS = 8;
constexpr int REORDER_NUMBER_OF_THREADS_PER_BLOCK = WARP_SIZE * REORDER_NUMBER_OF_WARPS;
constexpr int REORDER_NUMBER_OF_ITEM_PER_WARP = RADIX_SORT_BLOCK_SIZE / REORDER_NUMBER_OF_WARPS;
constexpr int REORDER_NUMBER_OF_ITEM_PER_THREAD = REORDER_NUMBER_OF_ITEM_PER_WARP / 32;

constexpr int LOOKBACK_TABLE_SIZE = 1024;
constexpr int MAX_LOOK_BACK = 64;
constexpr int TAIL_BITS = 5;
constexpr auto TAIL_MASK = 0xFFFFFFFFu << TAIL_BITS;
static_assert( MAX_LOOK_BACK < LOOKBACK_TABLE_SIZE, "" );

#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
#define ITS 1
#endif

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

using RADIX_SORT_VALUE_TYPE = u32;

#if defined( DESCENDING_ORDER )
constexpr u32 ORDER_MASK_32 = 0xFFFFFFFF;
constexpr u64 ORDER_MASK_64 = 0xFFFFFFFFFFFFFFFFllu;
#else
constexpr u32 ORDER_MASK_32 = 0;
constexpr u64 ORDER_MASK_64 = 0llu;
#endif

__device__ constexpr u32 div_round_up( u32 val, u32 divisor ) noexcept { return ( val + divisor - 1 ) / divisor; }

template<int NElement, int NThread, class T>
__device__ void clearShared( T* sMem, T value )
{
	for( int i = 0; i < NElement; i += NThread )
	{
		if( i < NElement )
		{
			sMem[i + threadIdx.x] = value;
		}
	}
}

__device__ inline u32 getKeyBits( u32 x ) { return x ^ ORDER_MASK_32; }
__device__ inline u64 getKeyBits( u64 x ) { return x ^ ORDER_MASK_64; }
__device__ inline u32 extractDigit( u32 x, u32 bitLocation ) { return ( x >> bitLocation ) & RADIX_MASK; }
__device__ inline u32 extractDigit( u64 x, u32 bitLocation ) { return (u32)( ( x >> bitLocation ) & RADIX_MASK ); }

template <class T>
__device__ inline T scanExclusive( T prefix, T* sMemIO, int nElement )
{
	// assert(nElement <= nThreads)
	bool active = threadIdx.x < nElement;
	T value = active ? sMemIO[threadIdx.x] : 0;
	T x = value;

	for( u32 offset = 1; offset < nElement; offset <<= 1 )
	{
		if( active && offset <= threadIdx.x )
		{
			x += sMemIO[threadIdx.x - offset];
		}

		__syncthreads();

		if( active )
		{
			sMemIO[threadIdx.x] = x;
		}

		__syncthreads();
	}

	T sum = sMemIO[nElement - 1];

	__syncthreads();

	if( active )
	{
		sMemIO[threadIdx.x] = x + prefix - value;
	}

	__syncthreads();

	return sum;
}

template <class RADIX_SORT_KEY_TYPE>
__device__ inline void onesweep_count( RADIX_SORT_KEY_TYPE* inputs, u32 numberOfInputs, u32* gpSumBuffer, u32 startBits, u32* counter )
{
	__shared__ u32 localCounters[sizeof( RADIX_SORT_KEY_TYPE )][BIN_SIZE];

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		for( int j = threadIdx.x; j < BIN_SIZE; j += ONESWEEP_COUNT_THREADS_PER_BLOCK )
		{
			localCounters[i][j] = 0;
		}
	}

	u32 numberOfBlocks = div_round_up( numberOfInputs, ONESWEEP_COUNT_ITEM_PER_BLOCK );
	__shared__ u32 iBlock;
	for(;;)
	{
		if( threadIdx.x == 0 )
		{
			iBlock = atomicInc( counter, 0xFFFFFFFF );
		}

		__syncthreads();

		if( numberOfBlocks <= iBlock )
			break;
    
		for( int j = 0; j < ONESWEEP_COUNT_ITEMS_PER_THREAD; j++ )
		{
			u32 itemIndex = iBlock * ONESWEEP_COUNT_ITEM_PER_BLOCK + threadIdx.x * ONESWEEP_COUNT_ITEMS_PER_THREAD + j;
			if( itemIndex < numberOfInputs )
			{
				auto item = inputs[itemIndex];
				for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
				{
					u32 bitLocation = startBits + i * N_RADIX;
					u32 bits = extractDigit( getKeyBits( item ), bitLocation );
					atomicInc( &localCounters[i][bits], 0xFFFFFFFF );
				}
			}
		}

		__syncthreads();
	}

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		scanExclusive<u32>( 0, &localCounters[i][0], BIN_SIZE );
	}

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		for( int j = threadIdx.x; j < BIN_SIZE; j += ONESWEEP_COUNT_THREADS_PER_BLOCK )
		{
			atomicAdd( &gpSumBuffer[BIN_SIZE * i + j], localCounters[i][j] );
		}
	}
}

extern "C" __global__ void onesweep_count32( u32* inputs, u32 numberOfInputs, u32* gpSumBuffer, u32 startBits, u32* counter )
{
    onesweep_count(inputs, numberOfInputs, gpSumBuffer, startBits, counter);
}
extern "C" __global__ void onesweep_count64( u64* inputs, u32 numberOfInputs, u32* gpSumBuffer, u32 startBits, u32* counter )
{
    onesweep_count(inputs, numberOfInputs, gpSumBuffer, startBits, counter);
}

template <bool keyPair, class RADIX_SORT_KEY_TYPE>
__device__ __forceinline__ void onesweep_reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, u32 numberOfInputs, u32* gpSumBuffer,
												  volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits, u32 iteration )
{
	__shared__ u32 pSum[BIN_SIZE];

	struct SMem
	{
		struct Phase1
		{
			u16 blockHistogram[BIN_SIZE];
			u16 lpSum[BIN_SIZE * REORDER_NUMBER_OF_WARPS];
		};
		struct Phase2
		{
			RADIX_SORT_KEY_TYPE elements[RADIX_SORT_BLOCK_SIZE];
		};
		struct Phase3
		{
			RADIX_SORT_VALUE_TYPE elements[RADIX_SORT_BLOCK_SIZE];
			u8 buckets[RADIX_SORT_BLOCK_SIZE];
		};

		union
		{
			Phase1 phase1;
			Phase2 phase2;
			Phase3 phase3;
		} u;
	};
	__shared__ SMem smem;

	u32 bitLocation = startBits + N_RADIX * iteration;
	u32 blockIndex = blockIdx.x;
	u32 numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

	clearShared<BIN_SIZE * REORDER_NUMBER_OF_WARPS, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u16>( smem.u.phase1.lpSum, 0 );

	__syncthreads();

	RADIX_SORT_KEY_TYPE keys[REORDER_NUMBER_OF_ITEM_PER_THREAD];
	u32 localOffsets[REORDER_NUMBER_OF_ITEM_PER_THREAD];

	int warp = threadIdx.x / WARP_SIZE;
	int lane = threadIdx.x % WARP_SIZE;

	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		if( itemIndex < numberOfInputs )
		{
			keys[k] = inputKeys[itemIndex];
		}
	}
	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );

		// check the attendees
		u32 broThreads =
#if defined( ITS )
			__ballot_sync( 0xFFFFFFFF,
#else
			__ballot(
#endif
						   itemIndex < numberOfInputs );

		for( int j = 0; j < N_RADIX; ++j )
		{
			u32 bit = ( bucketIndex >> j ) & 0x1;
			u32 difference = ( 0xFFFFFFFF * bit ) ^
#if defined( ITS )
								__ballot_sync( 0xFFFFFFFF, bit != 0 );
#else
								__ballot( bit != 0 );
#endif
			broThreads &= ~difference;
		}

		u32 lowerMask = ( 1u << lane ) - 1;
		auto digitCount = smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		localOffsets[k] = digitCount + __popc( broThreads & lowerMask );
		
#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
		u32 leaderIdx = __ffs( broThreads ) - 1;
		if( lane == leaderIdx )
		{
			smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp] = digitCount + __popc( broThreads );
		}
#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
	}

	__syncthreads();

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = 0;
		for( int warp = 0; warp < REORDER_NUMBER_OF_WARPS; warp++ )
		{
			s += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		}
		smem.u.phase1.blockHistogram[bucketIndex] = s;
	}

	enum
	{
		STATUS_X = 0,
		STATUS_A,
		STATUS_P,
	};
	struct ParitionID
	{
		u64 value : 32;
		u64 block : 30;
		u64 flag : 2;
	};
	auto asPartition = []( u64 x )
	{
		ParitionID pa;
		memcpy( &pa, &x, sizeof( ParitionID ) );
		return pa;
	};
	auto asU64 = []( ParitionID pa )
	{
		u64 x;
		memcpy( &x, &pa, sizeof( u64 ) );
		return x;
	};

	if( threadIdx.x == 0 && LOOKBACK_TABLE_SIZE <= blockIndex )
	{
		// Wait until blockIndex < tail - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE
		while( ( atomicAdd( tailIterator, 0 ) & TAIL_MASK ) - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE <= blockIndex )
			;
	}
	__syncthreads();

	// A workaround for buffer clear on each iterations.
	u32 iterationBits = 0x20000000 * ( iteration & 0x1 );

	for( int i = threadIdx.x; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = smem.u.phase1.blockHistogram[i];
		int pIndex = BIN_SIZE * ( blockIndex % LOOKBACK_TABLE_SIZE ) + i;

		{
			ParitionID pa;
			pa.value = s;
			pa.block = blockIndex | iterationBits;
			pa.flag = STATUS_A;
			lookBackBuffer[pIndex] = asU64( pa );
		}

		u32 gp = gpSumBuffer[iteration * BIN_SIZE + i];

		u32 p = 0;

		for( int iBlock = (int)blockIndex - 1; 0 <= iBlock; iBlock-- )
		{
			int lookbackIndex = BIN_SIZE * ( iBlock % LOOKBACK_TABLE_SIZE ) + i;
			ParitionID pa;

			// when you reach to the maximum, flag must be STATUS_P(=0b10). flagRequire = 0b10
			// Otherwise, flag can be STATUS_A(=0b01) or STATUS_P(=0b10) flagRequire = 0b11
			int flagRequire = MAX_LOOK_BACK == blockIndex - iBlock ? STATUS_P : STATUS_A | STATUS_P;

			do
			{
				pa = asPartition( lookBackBuffer[lookbackIndex] );
			} while( ( pa.flag & flagRequire ) == 0 || pa.block != ( iBlock | iterationBits ) );

			u32 value = pa.value;
			p += value;
			if( pa.flag == STATUS_P )
			{
				break;
			}
		}

		ParitionID pa;
		pa.value = p + s;
		pa.block = blockIndex | iterationBits;
		pa.flag = STATUS_P;
		lookBackBuffer[pIndex] = asU64( pa );

		// complete global output location
		u32 globalOutput = gp + p;
		pSum[i] = globalOutput;
	}

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		while( ( atomicAdd( tailIterator, 0 ) & TAIL_MASK ) != ( blockIndex & TAIL_MASK ) )
			;

		atomicInc( tailIterator, numberOfBlocks - 1 /* after the vary last item, it will be zero */ );
	}

	__syncthreads();

	u32 prefix = 0;
	for( int i = 0; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		prefix += scanExclusive<u16>( prefix, smem.u.phase1.blockHistogram + i, min( REORDER_NUMBER_OF_THREADS_PER_BLOCK, BIN_SIZE ) );
	}

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = smem.u.phase1.blockHistogram[bucketIndex];

		pSum[bucketIndex] -= s; // pre-substruct to avoid pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex] to calculate destinations

		for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
		{
			int index = bucketIndex * REORDER_NUMBER_OF_WARPS + w;
			u32 n = smem.u.phase1.lpSum[index];
			smem.u.phase1.lpSum[index] = s;
			s += n;
		}
	}

	__syncthreads();

	for( int k = 0; k < REORDER_NUMBER_OF_ITEM_PER_THREAD; k++ )
	{
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		localOffsets[k] += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
	}

	__syncthreads();

	for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		if( itemIndex < numberOfInputs )
		{
			smem.u.phase2.elements[localOffsets[k]] = keys[k];
		}
	}

	__syncthreads();

	for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
		if( itemIndex < numberOfInputs )
		{
			auto item = smem.u.phase2.elements[i];
			u32 bucketIndex = extractDigit( getKeyBits( item ), bitLocation );

			// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
			u32 dstIndex = pSum[bucketIndex] + i;
			outputKeys[dstIndex] = item;
		}
	}

	if ( keyPair )
	{
		__syncthreads();

		for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
			u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
			if( itemIndex < numberOfInputs )
			{
				smem.u.phase3.elements[localOffsets[k]] = inputValues[itemIndex];
				smem.u.phase3.buckets[localOffsets[k]] = bucketIndex;
			}
		}

		__syncthreads();

		for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
			if( itemIndex < numberOfInputs )
			{
				auto item       = smem.u.phase3.elements[i];
				u32 bucketIndex = smem.u.phase3.buckets[i];

				// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
				u32 dstIndex = pSum[bucketIndex] + i;
				outputValues[dstIndex] = item;
			}
		}
	}
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKey32( 
    u32* inputKeys, 
    u32* outputKeys, 
    u32 numberOfInputs, 
    u32* gpSumBuffer, 
    volatile u64* lookBackBuffer, 
    u32* tailIterator, 
    u32 startBits,
    u32 iteration )
{
	onesweep_reorder<false /*keyPair*/>( inputKeys, outputKeys, nullptr, nullptr, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKeyPair32( 
    u32* inputKeys, 
    u32* outputKeys, 
    RADIX_SORT_VALUE_TYPE* inputValues, 
    RADIX_SORT_VALUE_TYPE* outputValues,
    u32 numberOfInputs,
    u32* gpSumBuffer,
    volatile u64* lookBackBuffer, 
    u32* tailIterator, 
    u32 startBits,
    u32 iteration )
{
	onesweep_reorder<true /*keyPair*/>( inputKeys, outputKeys, inputValues, outputValues, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}

extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKey64( 
    u64* inputKeys, 
    u64* outputKeys, 
    u32 numberOfInputs, 
    u32* gpSumBuffer, 
    volatile u64* lookBackBuffer, 
    u32* tailIterator, 
    u32 startBits, 
    u32 iteration )
{
	onesweep_reorder<false /*keyPair*/>( inputKeys, outputKeys, nullptr, nullptr, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKeyPair64(
    u64* inputKeys, 
    u64* outputKeys, 
    RADIX_SORT_VALUE_TYPE* inputValues, 
    RADIX_SORT_VALUE_TYPE* outputValues,
    u32 numberOfInputs,
    u32* gpSumBuffer,
    volatile u64* lookBackBuffer, 
    u32* tailIterator, 
    u32 startBits, 
    u32 iteration )
{
	onesweep_reorder<true /*keyPair*/>( inputKeys, outputKeys, inputValues, outputValues, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}

)";

constexpr uint64_t div_round_up64(uint64_t val, uint64_t divisor) noexcept { return (val + divisor - 1) / divisor; }
constexpr uint64_t next_multiple64(uint64_t val, uint64_t divisor) noexcept { return div_round_up64(val, divisor) * divisor; }

struct KeyValueSoA32
{
	uint32_t* key;
	uint32_t* value;
};
struct KeyValueSoA64
{
	uint64_t* key;
	uint32_t* value;
};
class OnesweepSort
{
public:
	OnesweepSort( oroDevice device )
	{
		shaderCompile(device);
		allocate();
	}
	~OnesweepSort()
	{
		oroModuleUnload(m_module);
		oroFree(m_gpSumBuffer);
		oroFree(m_lookbackBuffer);
		oroFree(m_tailIterator);
		oroFree(m_gpSumCounter);
	}

	OnesweepSort(const OnesweepSort&) = delete;
	void operator=(const OnesweepSort&) = delete;

	void sort(const KeyValueSoA32& elementsToSort, const KeyValueSoA32& tmp, uint32_t n, int startBit, int endBit, oroStream stream)
	{
		sort(elementsToSort, tmp, n, startBit, endBit, stream, false /*is64bit*/);
	}
	void sort(const KeyValueSoA64& elementsToSort, const KeyValueSoA64& tmp, uint32_t n, int startBit, int endBit, oroStream stream)
	{
		sort(elementsToSort, tmp, n, startBit, endBit, stream, true /*is64bit*/);
	}
private:
	template <class KeyValueSoAT>
	void sort(const KeyValueSoAT& elementsToSort, const KeyValueSoAT& tmp, uint32_t n, int startBit, int endBit, oroStream stream, bool is64bit )
	{
		bool keyPair = elementsToSort.value != nullptr;

		int nIteration = div_round_up64(endBit - startBit, 8);
		uint64_t numberOfBlocks = div_round_up64(n, RADIX_SORT_BLOCK_SIZE);

		// Buffers
		oroMemsetD8Async(m_gpSumBuffer, 0, m_gpSumBufferBytes, stream);
		oroMemsetD8Async(m_lookbackBuffer, 0, m_lookBackBufferBytes, stream);
		oroMemsetD32Async(m_gpSumCounter, 0, 1, stream);
		oroMemsetD32Async(m_tailIterator, 0, 1, stream);

		FunctionSet funcset = is64bit ? m_func64 : m_func32;

		{
			const void* args[] = { &elementsToSort.key, &n, &m_gpSumBuffer, &startBit, &m_gpSumCounter };
			oroModuleLaunchKernel(funcset.count,
				funcset.blocksForCount, 1, 1,
				ONESWEEP_COUNT_THREADS_PER_BLOCK, 1, 1,
				0, /* shared */
				stream,
				args, 
				0 /* extra */);
		}


		auto s = elementsToSort;
		auto d = tmp;
		for (int i = 0; i < nIteration; i++)
		{
			if (keyPair)
			{
				const void* args[] = { &s.key, &d.key, &s.value, &d.value, &n, &m_gpSumBuffer, &m_lookbackBuffer, &m_tailIterator, &startBit, &i };
				oroModuleLaunchKernel(funcset.reorderKeyPair,
					numberOfBlocks, 1, 1,
					REORDER_NUMBER_OF_THREADS_PER_BLOCK, 1, 1,
					0, /* shared */
					stream,
					args,
					0 /* extra */);
			}
			else
			{
				const void* args[] = { &s.key, &d.key, &n, &m_gpSumBuffer, &m_lookbackBuffer, &m_tailIterator, &startBit, &i };
				oroModuleLaunchKernel(funcset.reorderKey,
					numberOfBlocks, 1, 1,
					REORDER_NUMBER_OF_THREADS_PER_BLOCK, 1, 1,
					0, /* shared */
					stream,
					args,
					0 /* extra */);
			}

			std::swap(s, d);
		}

		if (s.key /* current output */ != elementsToSort.key)
		{
			oroMemcpyDtoDAsync((oroDeviceptr)elementsToSort.key, (oroDeviceptr)tmp.key, sizeof(uint32_t) * n, stream);

			if (keyPair)
			{
				oroMemcpyDtoDAsync((oroDeviceptr)elementsToSort.value, (oroDeviceptr)tmp.value, sizeof(uint32_t) * n, stream);
			}
		}
	}
private:
	void shaderCompile(oroDevice device)
	{
		orortcProgram program = 0;
		orortcCreateProgram(&program, kernel, "onesweep", 0, 0, 0);

		std::vector<const char*> optionChars;
		orortcResult compileResult = orortcCompileProgram(program, optionChars.size(), optionChars.data());

		// print compilation log
		size_t logSize = 0;
		orortcGetProgramLogSize(program, &logSize);
		if (1 < logSize)
		{
			std::vector<char> compileLog(logSize);
			orortcGetProgramLog(program, compileLog.data());
			printf("%s", compileLog.data());
		}

		// get compiled code
		size_t codeSize = 0;
		orortcGetCodeSize(program, &codeSize);

		std::vector<char> codec(codeSize);
		orortcGetCode(program, codec.data());

		orortcDestroyProgram(&program);

		oroModuleLoadData(&m_module, codec.data());

		oroModuleGetFunction(&m_func32.count, m_module, "onesweep_count32");
		oroModuleGetFunction(&m_func32.reorderKey, m_module, "onesweep_reorderKey32");
		oroModuleGetFunction(&m_func32.reorderKeyPair, m_module, "onesweep_reorderKeyPair32");

		oroModuleGetFunction(&m_func64.count, m_module, "onesweep_count64");
		oroModuleGetFunction(&m_func64.reorderKey, m_module, "onesweep_reorderKey64");
		oroModuleGetFunction(&m_func64.reorderKeyPair, m_module, "onesweep_reorderKeyPair64");

		oroDeviceProp props = {};
		oroGetDeviceProperties(&props, device);
		{
			int maxBlocksPerMP = 0;
			oroError e = oroModuleOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMP, m_func32.count, ONESWEEP_COUNT_THREADS_PER_BLOCK, 0);
			m_func32.blocksForCount = e == oroSuccess ? maxBlocksPerMP * props.multiProcessorCount : 2048;
		}
		{
			int maxBlocksPerMP = 0;
			oroError e = oroModuleOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMP, m_func64.count, ONESWEEP_COUNT_THREADS_PER_BLOCK, 0);
			m_func64.blocksForCount = e == oroSuccess ? maxBlocksPerMP * props.multiProcessorCount : 2048;
		}
	}
	void allocate()
	{
		oroMalloc(&m_gpSumBuffer, m_gpSumBufferBytes);
		oroMalloc(&m_lookbackBuffer, m_lookBackBufferBytes);

		oroMalloc(&m_tailIterator, 4);
		oroMalloc(&m_gpSumCounter, 4);
	}

	oroModule m_module = 0;

	struct FunctionSet
	{
		int blocksForCount;
		oroFunction count;
		oroFunction reorderKey;
		oroFunction reorderKeyPair;
	};
	FunctionSet m_func32 = {};
	FunctionSet m_func64 = {};

	uint64_t m_gpSumBufferBytes = sizeof(uint32_t) * BIN_SIZE * sizeof(uint64_t /* key type (larger) */);
	uint64_t m_lookBackBufferBytes = sizeof(uint64_t) * (BIN_SIZE * LOOKBACK_TABLE_SIZE);

	void* m_gpSumBuffer = 0;
	void* m_lookbackBuffer = 0;
	void* m_tailIterator = 0;
	void* m_gpSumCounter = 0;
};
}