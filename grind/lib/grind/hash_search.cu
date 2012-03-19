#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <grind/log.h>

#include <vector_types.h>
#include <cutil_math.h>

#include "cuda_types.h"
#include "timer.h"
#include "algorithm/transform.h"




//---------------------------------------------------------------------------------------
#if 0
__host__ __device__
uint hashFn( 	int x,
				int y,
				int z,
				int sz )
{
	const uint p1 = 73856093;
	const uint p2 = 19349663;
	const uint p3 = 83492791;
	int result = p1 * x ^ p2 * y ^ p3 * z;
	result = abs( result ); // < mod of negative integer was returning incorrect results, particularly when converted to a uint
	result %= sz;
	return result;
}
#else
__host__ __device__
uint hashFn( 	int x,
				int y,
				int z,
				int sz )
{
	const uint p1 = 73856093;
	const uint p2 = 19349663;
	const uint p3 = 83492791;
	uint result = (p1 * x) ^ (p2 * y) ^ (p3 * z);
	result %= sz;
	return result;
}
#endif


//---------------------------------------------------------------------------------------
struct HashOpBase
{
	HashOpBase( float3 i_CellDim,
			int i_TableSize )
	{
		m_TableSize = i_TableSize;
		m_OneOverCellDim.x = 1.0f / i_CellDim.x;
		m_OneOverCellDim.y = 1.0f / i_CellDim.y;
		m_OneOverCellDim.z = 1.0f / i_CellDim.z;
	}

	int getTableSize() const { return m_TableSize; }

	__host__ __device__
	void getCellIndex( const float3& src,
			int& o_X,
			int& o_Y,
			int& o_Z )
	{
		o_X = int( floor( src.x * m_OneOverCellDim.x ) );
		o_Y = int( floor( src.y * m_OneOverCellDim.y ) );
		o_Z = int( floor( src.z * m_OneOverCellDim.z ) );
	}

protected:
	float3 m_OneOverCellDim;
	int m_TableSize;
};

//---------------------------------------------------------------------------------------
//! return the hash table index given an input point
struct HashValueOp
: public thrust::unary_function< float3, int >
, public HashOpBase
{
	HashValueOp( float3 i_CellDim,
				 int i_TableSize )
	: HashOpBase( i_CellDim, i_TableSize )
	{}

	__host__ __device__
	uint operator()( const float3& src )
	{
		int cx, cy, cz;
		getCellIndex( src, cx, cy, cz );

		return hashFn( cx, cy, cz, m_TableSize );
	}
};


//---------------------------------------------------------------------------------------
//! return the index of the nearest neighbour given a query point
struct HashNearestNeighbourOp
: public thrust::unary_function< float3, int >
, public HashOpBase
{
	HashNearestNeighbourOp( float3 i_CellDim,
							int i_TableSize,
							const float3* i_P,
							const int* i_PIndices,
							const int* i_HashVal,
							const int* i_HashIndices )
	: HashOpBase( i_CellDim, i_TableSize )
	, m_P( i_P )
	, m_PIndices( i_PIndices )
	, m_HashVal( i_HashVal )
	, m_HashIndices( i_HashIndices )
	{
	}

	__host__ __device__
	int operator()( const float3& i_QueryPt )
	{
		float best_dist_squared = 1.0f;
		int best_id = -1;

		int cx, cy, cz;
		getCellIndex( i_QueryPt, cx, cy, cz );

		// for each neighbouring cell
		for( int iz = -1; iz <= 1; ++iz ) {
			for( int iy = -1; iy <= 1; ++iy ) {
				for( int ix = -1; ix <= 1; ++ix ) {
					// get the hash id of the cell
					int neighbour_hash_id = hashFn( cx + ix, cy + iy, cz + iz, m_TableSize );
					int start_id = neighbour_hash_id > 0 ? m_HashIndices[ neighbour_hash_id-1 ] : 0;
					int end_id = m_HashIndices[ neighbour_hash_id ];

					for( int j = start_id; j < end_id; ++j ){
						int p_id = m_PIndices[ j ];
						const float3& candidateP  = *( m_P + p_id );
						float3 delta = i_QueryPt - candidateP;
						delta = delta * m_OneOverCellDim;
						float dist_squared = dot( delta, delta );
						if( dist_squared < best_dist_squared ){
							best_id = p_id;
							best_dist_squared = dist_squared;
						}
					}
				}
			}
		}
		return best_id;
	}

private:
	const float3* m_P;
	const int* m_PIndices;
	const int* m_HashVal;
	const int* m_HashIndices;
};


//---------------------------------------------------------------------------------------
// note don't try to pass a float3 across directly
extern "C"
void hashSearchBuildDevice( const float3* i_CellDim,
                            int i_HashTableSize,
                            int i_PSize,
                            const float3* i_P,
                            int* o_HashVal,
                            int* o_PIndices,
                            int* o_HashIndices )
{
#ifdef __DEVICE_EMULATION__
	const float3* ptr_P( i_P);
	int* ptr_HashVal( o_HashVal );
	int* ptr_PIndices( o_PIndices );
	int* ptr_HashIndices( o_HashIndices );
#else
	thrust::device_ptr< const float3 > ptr_P( i_P);
	thrust::device_ptr< int > ptr_HashVal( o_HashVal );
	thrust::device_ptr< int > ptr_PIndices( o_PIndices );
	thrust::device_ptr< int > ptr_HashIndices( o_HashIndices );
#endif

	HashValueOp hash_op( *i_CellDim, i_HashTableSize );

	// calc hash values for each p
	thrust::transform( ptr_P, ptr_P + i_PSize, ptr_HashVal, hash_op );

	// calc index for each p
	thrust::sequence( ptr_PIndices, ptr_PIndices + i_PSize );

	// sort indices based on hash value
	thrust::sort_by_key( ptr_HashVal, ptr_HashVal + i_PSize, ptr_PIndices );

	// calc index for each hash entry
	thrust::sequence( ptr_HashIndices, ptr_HashIndices + i_HashTableSize );

	// work out our magic table
	thrust::upper_bound( ptr_HashVal, ptr_HashVal + i_PSize, ptr_HashIndices, ptr_HashIndices + i_HashTableSize, ptr_HashIndices );
}

//---------------------------------------------------------------------------------------
extern "C"
void hashSearchFindNearestNeighbourDevice( const float3* i_CellDim,
                                           int i_HashTableSize,
                                           int i_PSize,
                                           const float3* i_P,
                                           const int* i_HashVal,
                                           const int* i_PIndices,
                                           const int* i_HashIndices,
                                           int i_QueryPCount,
                                           const float3* i_QueryP,
                                           int* o_ClosestIndices )
{
#ifdef __DEVICE_EMULATION__
	const float3* first( i_QueryP );
	const float3* last( i_QueryP + i_QueryPCount );
	int* ptr_ClosestIndices( o_ClosestIndices );
#else
	thrust::device_ptr< const float3 > first( i_QueryP );
	thrust::device_ptr< const float3 > last( i_QueryP + i_QueryPCount );
	thrust::device_ptr< int > ptr_ClosestIndices( o_ClosestIndices );
#endif

	HashNearestNeighbourOp nn_op( *i_CellDim, i_HashTableSize, i_P, i_PIndices, i_HashVal, i_HashIndices );

	GRIND_TRANSFORM( first, last, ptr_ClosestIndices, nn_op );
}
