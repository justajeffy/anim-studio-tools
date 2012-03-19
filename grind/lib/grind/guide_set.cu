/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: $"
 */

//-------------------------------------------------------------------------------------------------
#include <vector_types.h> // for float3 etc
#include <cutil_math.h>
#include <iostream>

#include <grind/algorithm/counting_iterator.h>
#include <grind/algorithm/for_each.h>


class SurfaceNormalGroomFunctor
{
public:
	__host__ __device__
	SurfaceNormalGroomFunctor( unsigned int _CurveCount,
			unsigned int _CvCount,
			const float3* _MeshP,
			const float3* _MeshN,
			const float* _SpanLength,
			float3* _GuideP,
			float3* _PrevP )
	: i_CurveCount( _CurveCount )
	, i_CvCount( _CvCount )
	, i_MeshP( _MeshP )
	, i_MeshN( _MeshN )
	, i_SpanLength( _SpanLength )
	, o_GuideP( _GuideP )
	, o_PrevP( _PrevP )
	{}

	__host__ __device__
	void operator()( int hair_id )
	{
		int cv_id = hair_id * i_CvCount;

		float3 p = i_MeshP[ hair_id ];
		float3 hair_direction = i_MeshN[ hair_id ];

		o_GuideP[ cv_id ] = p;
		o_PrevP[ cv_id ] = p;
		cv_id++;
		for ( int cv = 0 ; cv < i_CvCount - 1 ; ++cv, ++cv_id )
		{
			//p.y += 0.1f;
			p += hair_direction * i_SpanLength[ hair_id ];
			o_GuideP[ cv_id ] = p;
			o_PrevP[ cv_id ] = p;
		}
	}

private:
	unsigned int i_CurveCount;
	unsigned int i_CvCount;
	const float3* i_MeshP;
	const float3* i_MeshN;
	const float* i_SpanLength;
	float3* o_GuideP;
	float3* o_PrevP;
};

//-------------------------------------------------------------------------------------------------
extern "C"
void surfaceNormalGroomDevice( 	unsigned int i_CurveCount,
								unsigned int i_CvCount,
								const float3* i_MeshP,
								const float3* i_MeshN,
								const float* i_SpanLength,
								float3* o_GuideP,
								float3* o_PrevP )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_CurveCount );

	SurfaceNormalGroomFunctor f( i_CurveCount, i_CvCount, i_MeshP, i_MeshN, i_SpanLength, o_GuideP, o_PrevP );

	GRIND_FOR_EACH( first, last, f );
}

//-------------------------------------------------------------------------------------------------
class TangentSpaceUpdateFunctor
{
public:
	__host__ __device__
	TangentSpaceUpdateFunctor( 	unsigned int _CurveCount,
			unsigned int _CvCount,
			const float3* _GuidePTangent,
			const float3* _GuideAcrossTangent,
			const float3* _MeshP,
			const float3* _MeshN,
			const float3* _MeshT,
			float3* _GuideP,
			float3* _GuideAcross )
	: i_CurveCount( _CurveCount )
	, i_CvCount( _CvCount )
	, i_GuidePTangent( _GuidePTangent )
	, i_GuideAcrossTangent( _GuideAcrossTangent )
	, i_MeshP( _MeshP )
	, i_MeshN( _MeshN )
	, i_MeshT( _MeshT )
	, o_GuideP( _GuideP )
	, o_GuideAcross( _GuideAcross )
	{}

	__host__ __device__
	void operator()( int hair_id )
	{
		int cv_id = hair_id * i_CvCount;

		float3 root = i_MeshP[ hair_id ];
		float3 N = i_MeshN[ hair_id ];
		float3 T = i_MeshT[ hair_id ];
		float3 BN = cross( T,N );

		for ( int cv = 0 ; cv < i_CvCount ; ++cv, ++cv_id )
		{
			// the cv in tangent space
			float3 cvt = i_GuidePTangent[ cv_id ];

			o_GuideP[ cv_id ] = root + (T * cvt.x) + (N * cvt.y) + (BN * cvt.z);
		}

		float3 a = i_GuideAcrossTangent[ hair_id ];
		o_GuideAcross[ hair_id ] = (T * a.x) + (N * a.y) + (BN * a.z);
	}

private:
	unsigned int i_CurveCount;
	unsigned int i_CvCount;
	const float3* i_GuidePTangent;
	const float3* i_GuideAcrossTangent;
	const float3* i_MeshP;
	const float3* i_MeshN;
	const float3* i_MeshT;
	float3* o_GuideP;
	float3* o_GuideAcross;
};


//-------------------------------------------------------------------------------------------------
extern "C"
void tangentSpaceUpdateDevice( 	unsigned int i_CurveCount,
                    			unsigned int i_CvCount,
                    			const float3* i_GuidePTangent,
                    			const float3* i_GuideAcrossTangent,
                    			const float3* i_MeshP,
                    			const float3* i_MeshN,
                    			const float3* i_MeshT,
                    			float3* o_GuideP,
                    			float3* o_GuideAcross )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_CurveCount );

	TangentSpaceUpdateFunctor f( i_CurveCount, i_CvCount, i_GuidePTangent, i_GuideAcrossTangent, i_MeshP, i_MeshN, i_MeshT, o_GuideP, o_GuideAcross );

	GRIND_FOR_EACH( first, last, f );
}


//-------------------------------------------------------------------------------------------------
class GuideBuildAcrossDisplayFunctor
{
public:
	__host__ __device__
	GuideBuildAcrossDisplayFunctor( unsigned int _CurveCount,
                                    unsigned int _CvCount,
                                    const float3* _GuideP,
                                    const float3* _GuideAcross,
                                    float3* _GuideAcrossDisplay )
	: i_CurveCount( _CurveCount )
	, i_CvCount( _CvCount )
	, i_GuideP( _GuideP )
	, i_GuideAcross( _GuideAcross )
	, o_GuideAcrossDisplay( _GuideAcrossDisplay )
	{}

	__host__ __device__
	void operator()( int hair_id )
	{
		float3 root = i_GuideP[ hair_id * i_CvCount ];
		// make the across vector display scale based on the segment length of the guide
		float seg_length = length(  i_GuideP[ hair_id * i_CvCount + 1 ] - root );
		float3 across = root + i_GuideAcross[ hair_id ] * seg_length;

		o_GuideAcrossDisplay[ hair_id * 2 ] = root;
		o_GuideAcrossDisplay[ hair_id * 2 + 1 ] = across;
	}

private:
	unsigned int i_CurveCount;
	unsigned int i_CvCount;
	const float3* i_GuideP;
	const float3* i_GuideAcross;
	float3* o_GuideAcrossDisplay;
};

//-------------------------------------------------------------------------------------------------
extern "C"
void guideBuildAcrossDisplayDevice( unsigned int i_CurveCount,
                                    unsigned int i_CvCount,
                                    const float3* i_GuideP,
                                    const float3* i_GuideAcross,
                                    float3* o_GuideAcrossDisplay )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_CurveCount );

	GuideBuildAcrossDisplayFunctor f( i_CurveCount, i_CvCount, i_GuideP, i_GuideAcross, o_GuideAcrossDisplay );

	GRIND_FOR_EACH( first, last, f );
}

//-------------------------------------------------------------------------------------------------
struct CalcCurveLengthFunctor
{
public:
	__host__ __device__
	CalcCurveLengthFunctor( unsigned int _CvCount, const float3* _GuideP, float* _GuideLength )
	: i_CvCount( _CvCount ), i_GuideP( _GuideP ), o_GuideLength( _GuideLength )
	{}

	__host__ __device__
	void operator()( int i )
	{
		int cv_id = i * i_CvCount + 1;
		int cv_id_end = (i+1) * i_CvCount;
		float l = 0;
		for( ; cv_id != cv_id_end; ++cv_id )
		{
			float3 delta = i_GuideP[ cv_id ] - i_GuideP[ cv_id-1 ];
			l += length(delta);
		}
		o_GuideLength[i] = l;
	}

private:
	unsigned int i_CvCount;
	const float3* i_GuideP;
	float* o_GuideLength;
};

//-------------------------------------------------------------------------------------------------
extern "C"
void calcCurveLengthDevice( unsigned int i_CurveCount,
                            unsigned int i_CvCount,
                            const float3* i_CurveP,
                            float* o_CurveLength )
{
	grind::CountingIterator first( 0 );
	grind::CountingIterator last( i_CurveCount );

    CalcCurveLengthFunctor f( i_CvCount, i_CurveP, o_CurveLength );

    GRIND_FOR_EACH( first, last, f );

}

