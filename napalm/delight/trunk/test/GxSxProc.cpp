
#include "napalmDelight/GeoProcBase.h"
#include "napalmDelight/attr_helpers.h"
#include "napalmDelight/render.h"
#include "napalmDelight/SampleSx.h"

#include <napalm/core/exceptions.h>
#include <napalm/core/Table.h>
#include <napalm/core/TypedBuffer.h>

#include <ri.h>
#include <rx.h>

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
	boost::shared_ptr< SampleSx > m_surface;
	boost::shared_ptr< SampleSx > m_displacement;
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
		, m_surface( src.m_surface )
		, m_displacement( src.m_displacement )
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
		// only do this once the prim bounds have been hit
		n::object_table_ptr dispParams;
		if( m_userParams->getEntry( "displacement", dispParams ) ){
			m_displacement.reset( new SampleSx( dispParams, SX_RI_PARENT_CONTEXT ) );
		}

		n::object_table_ptr surfParams;
		if( m_userParams->getEntry( "surface", surfParams ) ){
			m_surface.reset( new SampleSx( surfParams, SX_RI_PARENT_CONTEXT ) );
		}

		// can be done in parallel (eg few prims with high face counds)
		for ( int face = 0 ; face < m_faceCount ; ++face )
		{
			RtBound bound;
			GxGetGeometryBound( getRawGeoHandle(), bound, "int face", &face, NULL );

			// pad bound (should be done more intelligently)
			float pad = 1.0f;
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
		SET_UP_BUFFER( t, N, V3fBuffer, np );
		SET_UP_BUFFER( t, s, FloatBuffer, np );
		SET_UP_BUFFER( t, t, FloatBuffer, np );
		SET_UP_BUFFER( t, Cs, V3fBuffer, np );
		SET_UP_BUFFER_FR( t, width, FloatBuffer, np );

		// temporary vars
		std::vector< Imath::V3f > dPdu, dPdv;

		bool has_P = evalSurf( pts, "P", t_P_buf_ptr );
		bool has_N = evalSurf( pts, "N", t_N_buf_ptr );
		bool has_s = evalSurf( pts, "s", t_s_buf_ptr );
		bool has_t = evalSurf( pts, "t", t_t_buf_ptr );
		bool has_Cs = evalSurf( pts, "Cs", t_Cs_buf_ptr );
		bool has_dPdu = evalSurf( pts, "dPdu", dPdu );
		bool has_dPdv = evalSurf( pts, "dPdv", dPdv );

		if( !has_P ) throw n::NapalmError( "couldn't eval P" );
		if( !has_N ) throw n::NapalmError( "couldn't eval N" );
		t_P_buf_ptr->getAttribs()->setEntry( "token", "P" );
		if( has_Cs ){
			t_Cs_buf_ptr->getAttribs()->setEntry( "token", "Cs" );
		}

		t_width_buf_ptr->getAttribs()->setEntry( "token", "varying float width" );

		if( m_displacement != NULL )
		{
			n::ObjectTable aovs;
			aovs.setEntry(0, "P" );

			n::ObjectTable ot;
			m_displacement->sample( t, aovs, ot );

			t["P"] = ot["P"];
			n::V3fBufferPtr pb;
			if( !t.getEntry( "P", pb ) )
				throw n::NapalmError( "failed to sample displacement" );

			// transform current -> object
			{
				n::V3fBuffer::w_type surf_P_fr = pb->rw();
				n::V3fBuffer::w_type::iterator surf_P_iter = surf_P_fr.begin();
				RxTransformPoints( "current", "object", surf_P_fr.size(), (RtPoint *)&surf_P_iter[0], 0.0f );
			}
			pb->getAttribs()->setEntry( "token", "P" );
		}

		if ( m_surface != NULL )
		{
			n::ObjectTable aovs;
			aovs.setEntry( 0, "Ci" );
			aovs.setEntry( 1, "my_radius" );

			n::ObjectTable ot;
			m_surface->sample( t, aovs, ot );

			t[ "Cs" ] = ot[ "Ci" ];
			n::V3fBufferPtr b;
			if( !t.getEntry( "Cs", b ) )
				throw n::NapalmError( "failed to sample surface shader" );
			b->getAttribs()->setEntry( "token", "Cs" );

			t[ "my_radius" ] = ot[ "my_radius" ];
		}

		n::FloatBufferPtr my_radius_buf_ptr;
		n::FloatBuffer::r_type my_radius_fr;
		n::FloatBuffer::r_type::iterator my_radius_iter;
		bool has_my_radius = t.getEntry( "my_radius", my_radius_buf_ptr );
		if( has_my_radius ){
			my_radius_fr = my_radius_buf_ptr->r();
			my_radius_iter = my_radius_fr.begin();
		}

		for( int i = 0; i < np; ++i, ++t_width_iter, ++my_radius_iter ){
			/* Ugly cross product. */
			float a = dPdu[i][1] * dPdv[i][2] - dPdu[i][2] * dPdv[i][1];
			float b = dPdu[i][2] * dPdv[i][0] - dPdu[i][0] * dPdv[i][2];
			float c = dPdu[i][0] * dPdv[i][1] - dPdu[i][1] * dPdv[i][0];
			/* Set width so the points cover about a quarter of the surface. */
			float radius = (0.5f / res) * sqrtf( sqrtf( a*a + b*b + c*c ) );
			if( has_my_radius )
				*t_width_iter = radius * (*my_radius_iter);
			else
				*t_width_iter = radius;
		}

//		if ( !has_Cs ) t.erase( "Cs" );
		t.erase( "s" );
		t.erase( "t" );

		napalm_delight::points( t );
	}
};


STANDARD_GEOPROCBASE_ENTRY( GeoEngine, GeoProcBase::PRIMITIVE_PROC );

