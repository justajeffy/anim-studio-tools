/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

//-------------------------------------------------------------------------------------------------
#include "cuda_types.h"
#include "log.h"
#include <cutil_math.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <grind/algorithm/counting_iterator.h>
#include <grind/algorithm/for_each.h>
#include <cfloat>


//-------------------------------------------------------------------------------------------------
class PerturbFunctor
{
public:
	__host__ __device__
	PerturbFunctor(	unsigned int _Seed,
					float _Deviation,
					float _OneOverGamma,
					unsigned int _RandomTableSize,
					float* _RandomTable,
					unsigned int _ResultSize,
					float* _Result )
	: i_Seed( _Seed )
	, i_Deviation( _Deviation )
	, i_OneOverGamma( _OneOverGamma )
	, i_RandomTableSize( _RandomTableSize )
	, i_ResultSize( _ResultSize )
	, o_Result( _Result )
	{}

	__host__ __device__
	void operator()( int i )
	{
		float v = o_Result[ i ];
		unsigned int random_id = ( i + i_Seed ) % i_RandomTableSize;
		float multiplier = 1.0f + ( powf(i_RandomTable[ random_id ],i_OneOverGamma) - 0.5f ) * 2.0f * i_Deviation;
		o_Result[ i ] = v * multiplier;
	}

private:

	unsigned int i_Seed;
	float i_Deviation;
	float i_OneOverGamma;
	unsigned int i_RandomTableSize;
	float* i_RandomTable;
	unsigned int i_ResultSize;
	float* o_Result;
};

//-------------------------------------------------------------------------------------------------
extern "C"
void perturbDevice( 	unsigned int i_Seed,
						float i_Deviation,
						float i_Gamma,
						unsigned int i_RandomTableSize,
						float* i_RandomTable,
						unsigned int i_ResultSize,
						float* o_Result )
{
	if( i_Gamma < 1.e-6f ) i_Gamma = 1.e-6f;
	float one_over_gamma = 1.0f/i_Gamma;

	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_ResultSize );

	PerturbFunctor f( i_Seed, i_Deviation, one_over_gamma, i_RandomTableSize, i_RandomTable, i_ResultSize, o_Result );

	GRIND_FOR_EACH( first, last, f );
}

//-------------------------------------------------------------------------------------------------
class ReproducablePerturbFunctor
{
public:
	__host__ __device__
	ReproducablePerturbFunctor( unsigned int _Seed,
								float _Deviation,
								float _OneOverGamma,
								unsigned int _RandomTableSize,
								const float* _RandomTable,
								unsigned int _ResultSize,
								const int* _Id,
								float* _Result )
	: i_Seed( _Seed )
	, i_Deviation( _Deviation )
	, i_OneOverGamma( _OneOverGamma )
	, i_RandomTableSize( _RandomTableSize )
	, i_RandomTable( _RandomTable )
	, i_ResultSize( _ResultSize )
	, i_Id( _Id )
	, o_Result( _Result )
	{}

	__host__ __device__
	void operator()( int i )
	{
		// work out index into random number table
		int random_id = ( i_Id[i] + i_Seed ) % i_RandomTableSize;

		float rand_val = i_RandomTable[ random_id ]; // 0-1
		float gamma_rand = powf( rand_val, i_OneOverGamma ); // 0-1
		float multiplier = 1.0f + ( gamma_rand - 0.5f ) * 2.0f * i_Deviation;

		o_Result[ i ] = o_Result[ i ] * multiplier;
	}

private:
	unsigned int i_Seed;
	float i_Deviation;
	float i_OneOverGamma;
	unsigned int i_RandomTableSize;
	const float* i_RandomTable;
	unsigned int i_ResultSize;
	const int* i_Id;
	float* o_Result;
};


//-------------------------------------------------------------------------------------------------
extern "C"
void
reproducablePerturbDevice( 	unsigned int i_Seed,
							float i_Deviation,
							float i_Gamma,
							unsigned int i_RandomTableSize,
							const float* i_RandomTable,
							unsigned int i_ResultSize,
							const int* i_Id,
							float* o_Result )
{
	if( i_Gamma < 1.e-6f ) i_Gamma = 1.e-6f;
	float one_over_gamma = 1.0f/i_Gamma;

	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_ResultSize );

	ReproducablePerturbFunctor f( i_Seed, i_Deviation, one_over_gamma, i_RandomTableSize, i_RandomTable, i_ResultSize, i_Id, o_Result );

	GRIND_FOR_EACH( first, last, f );
}

//-------------------------------------------------------------------------------------------------
struct ReproducableRandomSampleFunctor
{
	__host__ __device__
	ReproducableRandomSampleFunctor( int _Seed, int _Stride, int _RandomTableSize, const float* _RandomTable, const int* _Id, float* _Result )
	: i_Seed( _Seed )
	, i_Stride( _Stride )
	, i_RandomTableSize( _RandomTableSize )
	, i_RandomTable( _RandomTable )
	, i_Id( _Id )
	, o_Result( _Result )
	{}

	__host__ __device__
	void operator()( int i )
	{
		// work out index into random number table
		int random_id = ( i_Id[i] * i_Stride + i_Seed ) % i_RandomTableSize;
		float rand_val = i_RandomTable[ random_id ]; // 0-1
		o_Result[ i ] = rand_val;
	}

	int i_Seed;
	int i_Stride;
	int i_RandomTableSize;
	const float* i_RandomTable;
	const int* i_Id;
	float* o_Result;
};


//-------------------------------------------------------------------------------------------------
extern "C"
void
reproducableRandomSampleDevice( 	int i_Seed,
                                	int i_Stride,
                                	int i_RandomTableSize,
                                	const float* i_RandomTable,
                                	unsigned int i_ResultSize,
                                	const int* i_Id,
                                	float* o_Result )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_ResultSize );

	ReproducableRandomSampleFunctor f( i_Seed, i_Stride, i_RandomTableSize, i_RandomTable, i_Id, o_Result );

	GRIND_FOR_EACH( first, last, f );
}

//-------------------------------------------------------------------------------------------------
class RemapFunctor
{
public:
	__host__ __device__
	RemapFunctor(	float _Min,
                 	float _Max,
					unsigned int _ResultSize,
					float* _Result )
	: i_Min( _Min )
	, i_Max( _Max )
	, i_ResultSize( _ResultSize )
	, o_Result( _Result )
	{}

	__host__ __device__
	void operator()( int i )
	{
		float v = o_Result[ i ];
		v *= (i_Max-i_Min);
		v += i_Min;
		o_Result[ i ] = v;
	}

private:
	float i_Min;
	float i_Max;
	unsigned int i_ResultSize;
	float* o_Result;
};

//-------------------------------------------------------------------------------------------------
extern "C"
void remapDevice( 	float i_Min,
                  	float i_Max,
					unsigned int i_ResultSize,
					float* o_Result )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_ResultSize );

	RemapFunctor f( i_Min, i_Max, i_ResultSize, o_Result );

	GRIND_FOR_EACH( first, last, f );
}

//-------------------------------------------------------------------------------------------------
template< typename T >
void
compactTemplateDevice( size_t n, bool* i_Exists, size_t& o_ResultCount, T* o_Result )
{
#ifdef __DEVICE_EMULATION__
	T* a( o_Result );
	bool* s( i_Exists );
	T* a_end;
#else
	thrust::device_ptr< T > a( o_Result );
	thrust::device_ptr< bool > s( i_Exists );
	thrust::device_ptr< T > a_end;
#endif

	// remove if exists == 0
	a_end = thrust::remove_if( a, a+n, s, thrust::logical_not< bool >() );
	o_ResultCount = a_end - a;
}

//-------------------------------------------------------------------------------------------------
// can't template with extern "C"
extern "C"
void
compactFloatDevice(	size_t n,
					bool* i_Exists,
					size_t& o_ResultCount,
					float* o_Result )
{
	compactTemplateDevice( n, i_Exists, o_ResultCount, o_Result );
}


//-------------------------------------------------------------------------------------------------
// can't template with extern "C"
extern "C"
void
compactIntDevice(	size_t n,
                 	bool* i_Exists,
					size_t& o_ResultCount,
					int* o_Result )
{
	compactTemplateDevice( n, i_Exists, o_ResultCount, o_Result );
}

//-------------------------------------------------------------------------------------------------
extern "C"
void copyIntToFloatDevice( size_t n, int* i_Src, float* o_Dst )
{
#ifdef __DEVICE_EMULATION__
	int* src_ptr( i_Src );
	float* dst_ptr( o_Dst );
#else
	thrust::device_ptr<int> src_ptr( i_Src );
	thrust::device_ptr<float> dst_ptr( o_Dst );
#endif

	thrust::copy( src_ptr, src_ptr + n, dst_ptr );
}


//-------------------------------------------------------------------------------------------------
// problems using float3 for this thrust usage
struct MyFloat3
//: public float3
{
    float x, y, z;

    __host__ __device__
    MyFloat3() {}

    __host__ __device__
    MyFloat3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};

//-------------------------------------------------------------------------------------------------
// bounding box type
typedef thrust::pair<MyFloat3, MyFloat3> bbox;

//-------------------------------------------------------------------------------------------------
// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
struct bbox_reduction : public thrust::binary_function<bbox,bbox,bbox>
{
    __host__ __device__
    bbox operator()(bbox a, bbox b)
    {
        // lower left corner
        MyFloat3 ll( min(a.first.x, b.first.x)
                   , min(a.first.y, b.first.y)
                   , min(a.first.z, b.first.z) );

        // upper right corner
        MyFloat3 ur( max(a.second.x, b.second.x)
                   , max(a.second.y, b.second.y)
                   , max(a.second.z, b.second.z) );

        return bbox(ll, ur);
    }
};

//-------------------------------------------------------------------------------------------------
// convert a point to a bbox containing that point, (point) -> (point, point)
struct bbox_transformation : public thrust::unary_function<MyFloat3,bbox>
{
    __host__ __device__
    bbox operator()(MyFloat3 point)
    {
        return bbox(point, point);
    }
};

//-------------------------------------------------------------------------------------------------
template< typename ITER >
void boundingBoxAlgo( ITER first, ITER last, float3& o_BBoxMin, float3& o_BBoxMax )
{
	// start with a bounding box containing a single point
    bbox init( *first, *first );

    // transformation operation
    bbox_transformation unary_op;

    // binary reduction operation
    bbox_reduction binary_op;

    // compute the bounding box for the point set
    bbox result = thrust::transform_reduce( first, last, unary_op, init, binary_op);

    o_BBoxMin = make_float3( result.first.x, result.first.y, result.first.z );
    o_BBoxMax = make_float3( result.second.x, result.second.y, result.second.z );
}

//-------------------------------------------------------------------------------------------------
extern "C"
void
getBoundsDevice(	unsigned int i_Count,
                	float3* i_Data,
                	float3& o_BBoxMin,
                	float3& o_BBoxMax )
{
#ifdef __DEVICE_EMULATION__
	MyFloat3* first( (MyFloat3*)i_Data );
	MyFloat3* last( (MyFloat3*)i_Data + i_Count );
#else
	thrust::device_ptr<MyFloat3> first( (MyFloat3*)i_Data );
	thrust::device_ptr<MyFloat3> last( (MyFloat3*)i_Data + i_Count );
#endif

	boundingBoxAlgo( first, last, o_BBoxMin, o_BBoxMax );
}


//-------------------------------------------------------------------------------------------------
extern "C"
void
getIndexedBoundsDevice(	unsigned int i_Count,
                       	int* i_Indices,
                       	float3* i_Data,
                       	float3& o_BBoxMin,
                       	float3& o_BBoxMax )
{
#ifdef __DEVICE_EMULATION__
	typedef MyFloat3* ElementIterator;
	typedef int*	 IndexIterator;
#else
	typedef thrust::device_ptr<MyFloat3> ElementIterator;
	typedef thrust::device_ptr<int>		 IndexIterator;
#endif

	// set up permutation iterators to remap input points
	thrust::permutation_iterator< ElementIterator, IndexIterator > first( ElementIterator( (MyFloat3*)i_Data ), IndexIterator( i_Indices ) );
	thrust::permutation_iterator< ElementIterator, IndexIterator > last( ElementIterator( (MyFloat3*)i_Data ), IndexIterator( i_Indices + i_Count ) );

	boundingBoxAlgo( first, last, o_BBoxMin, o_BBoxMax );
}


namespace grind {

//-------------------------------------------------------------------------------------------------
template< typename T >
void setAllElementsDevice( const T& i_Val, unsigned int i_ResultSize, T* o_Result )
{
#ifdef __DEVICE_EMULATION__
	T* first( o_Result );
#else
	thrust::device_ptr< T > first( o_Result );
#endif

	thrust::fill( first, first + i_ResultSize, i_Val );
}

// explicit instantiations
template void setAllElementsDevice<int>( const int& i_Val, unsigned int i_ResultSize, int* o_Dst );
template void setAllElementsDevice<float>( const float& i_Val, unsigned int i_ResultSize, float* o_Dst );
template void setAllElementsDevice<Imath::V2f>( const Imath::V2f& i_Val, unsigned int i_ResultSize, Imath::V2f* o_Dst );
template void setAllElementsDevice<Imath::V3f>( const Imath::V3f& i_Val, unsigned int i_ResultSize, Imath::V3f* o_Dst );
template void setAllElementsDevice<Imath::V4f>( const Imath::V4f& i_Val, unsigned int i_ResultSize, Imath::V4f* o_Dst );

//-------------------------------------------------------------------------------------------------
template< typename T >
void getMaxValueDevice( size_t i_Count, T* i_Data, T& o_Result )
{
#ifdef __DEVICE_EMULATION__
	T* first( (T*)i_Data );
	T* last( (T*)i_Data + i_Count );
#else
	thrust::device_ptr<T> first( (T*)i_Data );
	thrust::device_ptr<T> last( (T*)i_Data + i_Count );
#endif
	o_Result = *( thrust::max_element( first, last ) );
}

// explicit instantiations
template void getMaxValueDevice<int>( size_t i_Count, int* i_Data, int& o_Result );
template void getMaxValueDevice<float>( size_t i_Count, float* i_Data, float& o_Result );

//-------------------------------------------------------------------------------------------------
template< typename T >
void getMinValueDevice( size_t i_Count, T* i_Data, T& o_Result )
{
#ifdef __DEVICE_EMULATION__
	T* first( (T*)i_Data );
	T* last( (T*)i_Data + i_Count );
#else
	thrust::device_ptr<T> first( (T*)i_Data );
	thrust::device_ptr<T> last( (T*)i_Data + i_Count );
#endif
	o_Result = *( thrust::min_element( first, last ) );
}

// explicit instantiations
template void getMinValueDevice<int>( size_t i_Count, int* i_Data, int& o_Result );
template void getMinValueDevice<float>( size_t i_Count, float* i_Data, float& o_Result );

//-------------------------------------------------------------------------------------------------
template< typename T >
void inclusiveScanDevice( size_t i_Count, const T* i_Data, T* o_Result )
{
#ifdef __DEVICE_EMULATION__
	T* first( (T*)i_Data );
	T* last( (T*)i_Data + i_Count );
	T* result_first( (T*)o_Result );
#else
	thrust::device_ptr<T> first( (T*)i_Data );
	thrust::device_ptr<T> last( (T*)i_Data + i_Count );
	thrust::device_ptr<T> result_first( (T*)o_Result );
#endif
	thrust::inclusive_scan( first, last, result_first );
}

// explicit instantiations
template void inclusiveScanDevice<int>( size_t i_Count, const int* i_Data, int* o_Result );
template void inclusiveScanDevice<float>( size_t i_Count, const float* i_Data, float* o_Result );

//-------------------------------------------------------------------------------------------------
template< typename T >
void adjacentDifferenceDevice( size_t i_Count, const T* i_Data, T* o_Result )
{
#ifdef __DEVICE_EMULATION__
	T* first( (T*)i_Data );
	T* last( (T*)i_Data + i_Count );
	T* result_first( (T*)o_Result );
#else
	thrust::device_ptr<T> first( (T*)i_Data );
	thrust::device_ptr<T> last( (T*)i_Data + i_Count );
	thrust::device_ptr<T> result_first( (T*)o_Result );
#endif
	thrust::adjacent_difference( first, last, result_first );
}

// explicit instantiations
template void adjacentDifferenceDevice<int>( size_t i_Count, const int* i_Data, int* o_Result );
template void adjacentDifferenceDevice<float>( size_t i_Count, const float* i_Data, float* o_Result );


//-------------------------------------------------------------------------------------------------
template< typename T >
int runLengthEncodeDevice( size_t i_Count, const T* i_Src, T* o_Dst, int* o_Lengths )
{
#ifdef __DEVICE_EMULATION__
	T* first( (T*)i_Src );
	T* last( (T*)i_Src + i_Count );
	T* dst( (T*)o_Dst );
	int* lengths( o_Lengths );
#else
	thrust::device_ptr<T> first( (T*)i_Src );
	thrust::device_ptr<T> last( (T*)i_Src + i_Count );
	thrust::device_ptr<T> dst( (T*)o_Dst );
	thrust::device_ptr<int> lengths( o_Lengths );
#endif

	int num_runs = thrust::reduce_by_key( first,
	                                      last,
	                                      thrust::constant_iterator<int>(1),
	                                      dst,
	                                      lengths ).first - dst;
	return num_runs;
}

// explicit instantiations
template int runLengthEncodeDevice<int>( size_t i_Count, const int* i_Src, int* o_Dst, int* o_Lengths );

//-------------------------------------------------------------------------------------------------
template< typename T >
T reduceDevice( size_t i_Count, const T* i_Data )
{
#ifdef __DEVICE_EMULATION__
	T* first( (T*)i_Data );
	T* last( (T*)i_Data + i_Count );
#else
	thrust::device_ptr<T> first( (T*)i_Data );
	thrust::device_ptr<T> last( (T*)i_Data + i_Count );
#endif
	return thrust::reduce( first, last );
}

// explicit instantiations
template int reduceDevice<int>( size_t i_Count, const int* i_Data );
template float reduceDevice<float>( size_t i_Count, const float* i_Data );

}
