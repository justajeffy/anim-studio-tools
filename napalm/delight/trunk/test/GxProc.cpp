
#include "napalmDelight/GeoProcBase.h"
#include "napalmDelight/attr_helpers.h"
#include "napalmDelight/render.h"

#include <napalm/core/exceptions.h>
#include <napalm/core/Table.h>
#include <napalm/core/TypedBuffer.h>

#include <ri.h>

namespace n = napalm;
using namespace napalm_delight;
using namespace drd;

#if defined( _WIN32 )
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

STANDARD_GEOPROCBASE_PREDEFINES();

#include "napalmDelight/buffer_helpers.h"

struct GeoEngine: public GeoProcBase
{
	//-------------------------------------------------------------------------------------------------
	GeoEngine( const RecursionLevel rl ) :
		GeoProcBase( rl )
	{
	}

	//-------------------------------------------------------------------------------------------------
	GeoEngine(	const GeoEngine& src,
				const RecursionLevel rl,
				int faceId ) :
		GeoProcBase( src, rl, faceId )
	{
	}

	//-------------------------------------------------------------------------------------------------
	~GeoEngine()
	{
	}

	//-------------------------------------------------------------------------------------------------
	void parse( const std::string& paramstr )
	{
		GeoProcBase::parse( paramstr );

		setAttrDefault( m_userParams, "count", 3 );
	}

	//-------------------------------------------------------------------------------------------------
	void emitPrimitiveProc( float detailSize )
	{
		RtBound bound;
		getPrimitiveBound( bound );

		GeoEngine* sub = new GeoEngine( *this, FACE_PROC, -1 );

		optionallyRenderPrimitiveBound( bound );
		RiProcedural( (RtPointer*) ( sub ), bound, Subdivide, Free );
	}

	//-------------------------------------------------------------------------------------------------
	void emitFaceProcs( float detailSize )
	{
		// can be done in parallel (eg few prims with high face counds)
		for ( int face = 0 ; face < m_faceCount ; ++face )
		{
			RtBound bound;
			GxGetGeometryBound( getRawGeoHandle(), bound, "int face", &face, NULL );

			// pad bound (should be done more intelligently)
			float pad = 0.2f;
			bound[0] -= pad;
			bound[1] += pad;
			bound[2] -= pad;
			bound[3] += pad;
			bound[4] -= pad;
			bound[5] += pad;

			GeoEngine* sub = new GeoEngine( *this, GEO_PROC, face );

			optionallyRenderFaceBound( bound );
			RiProcedural( (RtPointer*) ( sub ), bound, Subdivide, Free );
		}
	}

	//-------------------------------------------------------------------------------------------------
	void emitGeo( float detailSize )
	{
		int res = getAttr< int > ( m_userParams, "count" );
		int np = res * res;

		GxSurfacePoints pts = GxCreateSurfacePoints( np );

		for ( int v = 0 ; v < res ; ++v )
		{
			for ( int u = 0 ; u < res ; ++u )
			{
				float time = 0.0f;
				int ret =
						GxCreateSurfacePoint( m_geoHandle, m_faceId, ( u + 0.5f ) * 1.0f / res, ( v + 0.5f ) * 1.0f / res, time, v * res + u, pts );
				if ( ret != RIE_NOERROR ) throw n::NapalmError( "error creating surface points" );
			}
		}

		n::ObjectTable t;
		SET_UP_BUFFER( t, P, V3fBuffer, np );
		SET_UP_BUFFER( t, Cs, V3fBuffer, np );
		SET_UP_BUFFER_FR( t, width, FloatBuffer, np );

		// temporary vars
		std::vector< Imath::V3f > dPdu, dPdv;

		bool has_P = evalSurf( pts, "P", t_P_buf_ptr );
		bool has_Cs = evalSurf( pts, "Cs", t_Cs_buf_ptr );
		bool has_dPdu = evalSurf( pts, "dPdu", dPdu );
		bool has_dPdv = evalSurf( pts, "dPdv", dPdv );

		if( !has_P ) throw n::NapalmError( "couldn't eval P" );
		t_P_buf_ptr->getAttribs()->setEntry( "token", "P" );
		if( has_Cs ){
			t_Cs_buf_ptr->getAttribs()->setEntry( "token", "Cs" );
		}

		t_width_buf_ptr->getAttribs()->setEntry( "token", "varying float width" );

		for( int i = 0; i < np; ++i, ++t_width_iter ){
			/* Ugly cross product. */
			float a = dPdu[i][1] * dPdv[i][2] - dPdu[i][2] * dPdv[i][1];
			float b = dPdu[i][2] * dPdv[i][0] - dPdu[i][0] * dPdv[i][2];
			float c = dPdu[i][0] * dPdv[i][1] - dPdu[i][1] * dPdv[i][0];
			/* Set width so the points cover about a quarter of the surface. */
			*t_width_iter = (0.5f / res) * sqrtf( sqrtf( a*a + b*b + c*c ) );
		}

		if ( !has_Cs ) t.erase( "Cs" );

		napalm_delight::points( t );
	}
};


STANDARD_GEOPROCBASE_ENTRY( GeoEngine, GeoProcBase::PRIMITIVE_PROC );

