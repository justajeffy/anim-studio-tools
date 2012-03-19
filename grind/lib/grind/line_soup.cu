/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

//-------------------------------------------------------------------------------------------------
#include "cuda_types.h"
#include "log.h"
#include "cutil_math.h"
#include <grind/algorithm/for_each.h>
#include <grind/algorithm/counting_iterator.h>
#include <thrust/sort.h>


//-------------------------------------------------------------------------------------------------
class CalcKeyValueFunctor
{
public:
	__host__ __device__
	CalcKeyValueFunctor( 	unsigned int _LineSegCount,
							float _Vx,
							float _Vy,
							float _Vz,
							const float3* _P,
							bool _FurIsQuads,
							const uint* _Indices,
							float* _Keys,
							uint* _Vals )
	: i_LineSegCount( _LineSegCount )
	, i_Vx( _Vx )
	, i_Vy( _Vy )
	, i_Vz( _Vz )
	, i_P( _P )
	, i_FurIsQuads( _FurIsQuads )
	, i_Indices( _Indices )
	, o_Keys( _Keys )
	, o_Vals( _Vals )
	{}

	__host__ __device__
	void operator()( int line_id )
	{
		float3 view_pos = make_float3( i_Vx, i_Vy, i_Vz );

		// use centre of primitive
		uint ida, idb;

		if( i_FurIsQuads ){
			// diagonal centre of quad
			ida = i_Indices[ line_id * 4 ];
			idb = i_Indices[ line_id * 4 + 2 ];
		} else {
			// centre of line seg
			ida = i_Indices[ line_id * 2 ];
			idb = i_Indices[ line_id * 2 + 1 ];
		}

		float3 pa = i_P[ ida ];
		float3 pb = i_P[ idb ];

		float3 p_mid = ( pa + pb ) * 0.5f;

		float3 delta = view_pos - p_mid;

		// key is distance squared from eye position to midpoint of line segment
		o_Keys[ line_id ] = length( delta );

		// value is line segment id
		o_Vals[ line_id ] = ida;
	}

private:
	unsigned int i_LineSegCount;
	float i_Vx;
	float i_Vy;
	float i_Vz;
	const float3* i_P;
	bool i_FurIsQuads;
	const uint* i_Indices;
	float* o_Keys;
	uint* o_Vals;
};

//-------------------------------------------------------------------------------------------------
class ReformIndicesFunctor
{
public:
	__host__ __device__
	ReformIndicesFunctor( 	unsigned int _LineSegCount,
                          	bool _FurIsQuads,
                          	uint _IndexMidPt,
							const uint* _Vals,
							uint* _Indices )
	: i_LineSegCount( _LineSegCount )
	, i_FurIsQuads( _FurIsQuads )
	, i_IndexMidPt( _IndexMidPt )
	, i_Vals( _Vals )
	, o_Indices( _Indices )
	{}

	__host__ __device__
	void operator()( int seg_id )
	{
		// the base vert id
		uint id = i_Vals[ i_LineSegCount - seg_id - 1 ];

		if( i_FurIsQuads ){
			o_Indices[ seg_id * 4 + 0 ] = id;
			o_Indices[ seg_id * 4 + 1 ] = id + 1;
			o_Indices[ seg_id * 4 + 2 ] = id + 1 + i_IndexMidPt;
			o_Indices[ seg_id * 4 + 3 ] = id + i_IndexMidPt;
		} else {
			o_Indices[ seg_id * 2 + 0 ] = id;
			o_Indices[ seg_id * 2 + 1 ] = id + 1;
		}
	}

private:
	unsigned int i_LineSegCount;
	bool i_FurIsQuads;
	uint i_IndexMidPt;
	const uint* i_Vals;
	uint* o_Indices;
};

//-------------------------------------------------------------------------------------------------
extern "C"
void sortLineSoupIndicesDevice( 	unsigned int i_LineSegCount,
									float i_Vx,
									float i_Vy,
									float i_Vz,
									const float3* i_P,
									bool i_FurIsQuads,
									uint i_IndexMidPt,
									float* o_Keys,
									uint* o_Vals,
									uint* i_Indices )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_LineSegCount );

#ifdef __DEVICE_EMULATION__
	float* key_ptr( o_Keys );
	uint* val_ptr( o_Vals );
#else
	thrust::device_ptr< float > key_ptr( o_Keys );
	thrust::device_ptr< uint > val_ptr( o_Vals );
#endif

	CalcKeyValueFunctor f( i_LineSegCount, i_Vx,i_Vy,i_Vz, i_P, i_FurIsQuads, i_Indices, o_Keys, o_Vals );
	GRIND_FOR_EACH( first, last, f );

	thrust::sort_by_key( key_ptr, key_ptr + i_LineSegCount, val_ptr );

	ReformIndicesFunctor f2( i_LineSegCount, i_FurIsQuads, i_IndexMidPt, o_Vals, i_Indices );
	GRIND_FOR_EACH( first, last, f2 );
}


//-------------------------------------------------------------------------------------------------
class BuildQuadWidthFunctor
{
public:
	__host__ __device__
	BuildQuadWidthFunctor(	unsigned int _CurveCount,
							unsigned int _CvCount,
							float _Vx,
							float _Vy,
							float _Vz,
							const float* _Width,
							float3* _P,
							float3* _Norm )
	: i_CurveCount( _CurveCount )
	, i_CvCount( _CvCount )
	, i_Vx( _Vx )
	, i_Vy( _Vy )
	, i_Vz( _Vz )
	, i_Width( _Width )
	, o_P( _P )
	, o_Norm( _Norm )
	{}

	__host__ __device__
	void operator()( int fur_curve_id )
	{
		unsigned int n = i_CurveCount * i_CvCount;
		int fur_cv_id = fur_curve_id * i_CvCount;

		float3 view_dir = o_P[ fur_cv_id ] - make_float3( i_Vx, i_Vy, i_Vz );

		// need to keep this from previous cv so we don't skew result
		float3 central_p;

		for( int cv = 0; cv < i_CvCount; ++cv, ++fur_cv_id )
		{
			float3 tangent;
			if ( cv == 0 )
			{
				// forwards difference
				tangent = o_P[ fur_cv_id + 1 ] - o_P[ fur_cv_id ];
			}
			else if ( cv == i_CvCount - 1 )
			{
				// backwards difference
				tangent = o_P[ fur_cv_id ] - central_p;
			}
			else
			{
				// central difference
				tangent = o_P[ fur_cv_id + 1 ] - central_p;
			}

			float3 n_vec = o_Norm[ fur_cv_id ];
			//n_vec = view_dir;
			//float3 across_vec = cross( tangent, n_vec );
			float3 across_vec = normalize( cross( n_vec, tangent ) );

			float3 delta = across_vec * i_Width[fur_cv_id] * 0.5f;
			central_p = o_P[ fur_cv_id ];

			o_P[ fur_cv_id ] = central_p - delta;
			o_P[ fur_cv_id + n ] = central_p + delta;
			o_Norm[ fur_cv_id + n ] = n_vec;
		}
	}

private:
	unsigned int i_CurveCount;
	unsigned int i_CvCount;
	float i_Vx;
	float i_Vy;
	float i_Vz;
	const float* i_Width;
	float3* o_P;
	float3* o_Norm;
};


//-------------------------------------------------------------------------------------------------
extern "C"
void buildQuadWidthDevice( 	unsigned int i_CurveCount,
							unsigned int i_CvCount,
							float i_Vx,
							float i_Vy,
							float i_Vz,
							const float* i_Width,
							float3* o_P,
							float3* o_Norm )
{
	grind::CountingIterator first(0);
	grind::CountingIterator last(i_CurveCount);

	BuildQuadWidthFunctor f( i_CurveCount, i_CvCount, i_Vx, i_Vy, i_Vz, i_Width, o_P, o_Norm );

	GRIND_FOR_EACH( first, last, f );
}


//-------------------------------------------------------------------------------------------------
class BuildQuadUVWFunctor
{
public:
	__host__ __device__
	BuildQuadUVWFunctor(	unsigned int _CurveCount,
							unsigned int _CvCount,
							float4* _UVW )
	: i_CurveCount( _CurveCount )
	, i_CvCount( _CvCount )
	, o_UVW( _UVW )
	{}

	__host__ __device__
	void operator()( int fur_curve_id )
	{
		unsigned int n = i_CurveCount * i_CvCount;
		int fur_cv_id = fur_curve_id * i_CvCount;

		for( int cv = 0; cv < i_CvCount; ++cv, ++fur_cv_id )
		{
			float4 uvw = o_UVW[ fur_cv_id ];
			uvw.w = 0;
			o_UVW[ fur_cv_id ] = uvw;
			uvw.w = 1;
			o_UVW[ fur_cv_id + n ] = uvw;
		}
	}

private:
	unsigned int i_CurveCount;
	unsigned int i_CvCount;
	float4* o_UVW;
};

//-------------------------------------------------------------------------------------------------
extern "C"
void buildQuadUVWDevice( 	unsigned int i_CurveCount,
							unsigned int i_CvCount,
							float4* o_UVW )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_CurveCount );

	BuildQuadUVWFunctor f( i_CurveCount, i_CvCount, o_UVW );

	GRIND_FOR_EACH( first, last, f );
}

