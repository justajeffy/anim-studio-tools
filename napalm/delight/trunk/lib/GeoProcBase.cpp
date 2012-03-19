#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:GeoProcBase" );

#include "napalmDelight/GeoProcBase.h"
#include "napalmDelight/attr_helpers.h"
#include "napalmDelight/type_conversion.h"
#include "napalmDelight/exceptions.h"
#include "napalmDelight/render.h"

#include <napalm/core/Table.h>
#include <napalm/core/TypedBuffer.h>
#include <napalm/parsing/parsePythonDict.h>

#include <delight_box_helper.h>

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

#include <delightUtils/RxWrappers.h>

#include <rx.h>

#include <boost/foreach.hpp>

using namespace napalm_delight;
using namespace napalm;
using namespace drd;
using namespace Imath;

#define GEOPROCBASE_RENDER_PRIMITIVE_BOUNDS "renderPrimitiveBound"
#define GEOPROCBASE_RENDER_FACE_BOUNDS "renderFaceBound"
#define GEOPROCBASE_BOUNDS_OPACITY "boundOpacity"

//-------------------------------------------------------------------------------------------------
GeoProcBase::GeoProcBase( const RecursionLevel rl )
: m_userParams( new ObjectTable() )
, m_recursionLevel( rl )
, m_faceId( -1 )
, m_faceCount( 0 )
{}

//-------------------------------------------------------------------------------------------------
GeoProcBase::GeoProcBase( const GeoProcBase& src, const RecursionLevel rl, int faceId )
: m_geoHandle( src.m_geoHandle )
, m_recursionLevel( rl )
, m_faceId( faceId )
, m_faceCount( src.m_faceCount )
, m_timeSamples( src.m_timeSamples )
, m_offsetTimeSamples( src.m_offsetTimeSamples )
{
	// we won't be modifying this per-instance
	// so we'll share the table rather than clone
	m_userParams = src.m_userParams;
}

//-------------------------------------------------------------------------------------------------
GeoProcBase::~GeoProcBase()
{}

//-------------------------------------------------------------------------------------------------
float GeoProcBase::getCrudeBoundPad() const
{
	float boundPad = getAttr< float > ( m_userParams, "globalBoundPad" )
	        	   + getAttr< float > ( m_userParams, "displacementBound" );

	return boundPad;
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::getPrimitiveBound( RtBound result )
{
	GxGetGeometryBound( getRawGeoHandle(), result, NULL );

	float boundPad = getCrudeBoundPad();

	DRD_LOG_DEBUG( L, "padding geo bound by: " << boundPad << " (globalBoundPad + displacement bound)" );

	pad( result, boundPad );
}

//-------------------------------------------------------------------------------------------------
::GxGeometryHandle GeoProcBase::getRawGeoHandle()
{
	return *(m_geoHandle->get_data().get());
}


//-------------------------------------------------------------------------------------------------
void GeoProcBase::setUpGeoHandle()
{
	std::string geoName = getAttr< std::string > ( m_userParams, "geoName" );
	m_geoHandle = GxGetGeometryFromProcedural( geoName );
	m_faceCount = GxGetFaceCount( m_geoHandle );
	if ( m_faceCount > 0 )
	{
		DRD_LOG_DEBUG( L, "found '" << geoName << "' (" << m_faceCount << " faces)" );
	}
	else
	{
		throw NapalmDelightError( std::string( "failed to find geo named '" ) + geoName + "'" );
	}
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::subdivide( RtFloat detailSize )
{
//	DRD_LOG_DEBUG( L, "recursion level: " << m_recursionLevel );
	switch( m_recursionLevel ){
		case PRIMITIVE_PROC: emitPrimitiveProc( detailSize ); break;
		case FACE_PROC: emitFaceProcs( detailSize ); break;
		case GEO_PROC: emitGeo( detailSize ); break;
		default: throw NapalmDelightError( "undefined subdivision level" );
	}
}


//-------------------------------------------------------------------------------------------------
bool GeoProcBase::evalSurf( const GxSurfacePoints& pts, const char* name, std::vector<Imath::V3f>& result )
{
	unsigned int np = pts->get_numPoints();
	result.clear();
	result.resize( np, Imath::V3f( 0, 0, 0 ) );
	bool success = RIE_NOERROR == ::GxEvaluateSurface( np, pts->get_data(), name, 3, reinterpret_cast<float*>(&(result[0])));
	return success;
}

//-------------------------------------------------------------------------------------------------
bool GeoProcBase::evalSurf( const GxSurfacePoints& pts, const char* name, std::vector<float>& result )
{
	unsigned int np = pts->get_numPoints();
	result.clear();
	result.resize( np, 0.0f );
	bool success = RIE_NOERROR == ::GxEvaluateSurface( np, pts->get_data(), name, 1, reinterpret_cast<float*>(&(result[0])));
	return success;
}

//-------------------------------------------------------------------------------------------------
bool GeoProcBase::evalSurf( const drd::GxSurfacePoints& pts, const char* name, napalm::V3fBufferPtr result )
{
	unsigned int np = pts->get_numPoints();
	result->resize( np );

	napalm::V3fBuffer::w_type fr = result->rw();
	bool success = RIE_NOERROR == ::GxEvaluateSurface( np, pts->get_data(), name, 3, reinterpret_cast<float*>(&(fr[0])));
	return success;
}

//-------------------------------------------------------------------------------------------------
bool GeoProcBase::evalSurf( const drd::GxSurfacePoints& pts, const char* name, napalm::FloatBufferPtr result )
{
	unsigned int np = pts->get_numPoints();
	result->resize( np );

	napalm::FloatBuffer::w_type fr = result->rw();
	bool success = RIE_NOERROR == ::GxEvaluateSurface( np, pts->get_data(), name, 1, reinterpret_cast<float*>(&(fr[0])));
	return success;
}

//-------------------------------------------------------------------------------------------------
float GeoProcBase::calcCurveLength( int samples )
{
	GxSurfacePoints pts = GxCreateSurfacePoints( samples );

	const float time = 0.0f;

	// approximate with n samples
	for ( int i = 0 ; i < samples ; ++i )
	{
		float v = float( i ) / ( samples - 1 );
		GxCreateSurfacePoint( m_geoHandle, m_faceId, 0.5, v, time, i, pts );
	}

	std::vector< Imath::V3f > P;
	if( evalSurf( pts, "Pref", P ) ){
		DRD_LOG_DEBUG( L, "using Pref for curve length calculation" );
	} else {
		DRD_LOG_DEBUG( L, "using P for curve length calculation" );
		evalSurf( pts, "P", P );
	}
	float length = 0;
	for ( int i = 1 ; i < samples ; ++i )
	{
		length += ( P[ i - 1 ] - P[ i ] ).length();
	}
	return length;
}

//-------------------------------------------------------------------------------------------------
float GeoProcBase::calcSubFaceAreas( unsigned int i_faceId, int uSegs, int vSegs, std::vector< float >* areas )
{
	float area = 0.0f;

	if( areas ){
		areas->clear();
		areas->reserve( uSegs * vSegs );
	}

	int nu = uSegs + 1;
	int nv = vSegs + 1;

	GxSurfacePoints pts = GxCreateSurfacePoints( nu * nv );
	int i = 0;
	for ( int v = 0 ; v < nv ; ++v )
	{
		for ( int u = 0 ; u < nu ; ++u )
		{
			GxCreateSurfacePoint( m_geoHandle, i_faceId, float( u ) / ( nu - 1 ), float( v ) / ( nv - 1 ), 0, i++, pts );
		}
	}
	assert( i == nu * nv );

	std::vector< Imath::V3f > P;
	if( evalSurf( pts, "Pref", P ) ){
		DRD_LOG_DEBUG( L, "using Pref for face area calculation" );
	} else {
		DRD_LOG_DEBUG( L, "using P for face area calculation" );
		evalSurf( pts, "P", P );
	}
	assert( P.size() == nu * nv );

	for ( int v = 0 ; v < (nv-1) ; ++v )
	{
		for ( int u = 0 ; u < (nu-1) ; ++u )
		{
			float subFaceArea = 0;

			// b---c
			// | / |   <-- quad as 2 triangles
			// a---d

			// indices of sub-face corners
			int ai = v * nv + u;
			int bi = (v+1) * nv + u;
			int ci = (v+1) * nv + u + 1;
			int di = v * nv + u + 1;

			assert( ai < P.size() );
			assert( bi < P.size() );
			assert( ci < P.size() );
			assert( di < P.size() );

			// area of first triangle
			Imath::V3f ab = P[ ai ] - P[ bi ];
			Imath::V3f cb = P[ ci ] - P[ bi ];
			subFaceArea += 0.5 * ( ab.cross( cb ) ).length();

			// area of second triangle
			Imath::V3f cd = P[ ci ] - P[ di ];
			Imath::V3f ad = P[ ai ] - P[ di ];
			subFaceArea += 0.5 * ( cd.cross( ad ) ).length();

			if( areas )
				areas->push_back( subFaceArea );

			area += subFaceArea;
		}
	}

	if( areas ){
		assert( areas->size() == uSegs * vSegs );
	}

	return area;
}

//-------------------------------------------------------------------------------------------------
float GeoProcBase::calcFaceArea( unsigned int i_faceId )
{
	int segs =  getAttr< int > ( m_userParams, "areaApproximation" );
	DRD_LOG_DEBUG( L, "approximating area with " << segs << " segments" );

	return calcSubFaceAreas( i_faceId, segs, segs );
}

//-------------------------------------------------------------------------------------------------
Imath::V3f GeoProcBase::getCameraPos()
{
	Imath::V3f queryP(0,0,0);
	bool success = (RIE_NOERROR == RxTransformPoints( "camera", "object", 1, (RtPoint*) &queryP.x, 0.0f ) );
	if( !success )
	{
		throw NapalmDelightError( "Failed to get camera position" );
	}
	return queryP;
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::parse( const std::string& paramstr )
{
	try {
		DRD_LOG_DEBUG( L, "parsing " << paramstr );
		m_userParams = parsePythonDict( paramstr );

		setAttrDefault( m_userParams, "geoName", "" );
		setAttrDefault( m_userParams, "density", 200.0f );
#if 0
		// DON'T SPECIFY A DEFAULT FOR THIS, TD'S SHOULD BE DOING SO
		setAttrDefault( m_userParams, "globalBoundPad", 0.0f );
#endif

		// precision of area calculation
		setAttrDefault( m_userParams, "areaApproximation", 1 );

		// motion related (1 motion sample == no blur)
		setAttrDefault( m_userParams, "motionSamples", 1 );
		if( getAttr< int > ( m_userParams, "motionSamples" ) < 1 )
			throw NapalmDelightError( "invalid value for motion samples" );

		// depending on the rib configuration, samples to Gx may or may not
		// have to be offset by the rib's shutter offset
		setAttrDefault( m_userParams, "shutterOffsetMode", 0 );

		// displacement related
		float displacementBound = 0.0f;
		getRxAttribute( "displacementbound:sphere", displacementBound );
		m_userParams->setEntry( "displacementBound", displacementBound );
		DRD_LOG_INFO( L, "globalBoundPad: " << getAttr< float > ( m_userParams, "globalBoundPad" ) );
		DRD_LOG_INFO( L, "displacement bound: " << displacementBound );

		// bounds related
		setAttrDefault( m_userParams, GEOPROCBASE_RENDER_PRIMITIVE_BOUNDS, 0 );
		setAttrDefault( m_userParams, GEOPROCBASE_RENDER_FACE_BOUNDS, 0 );
		setAttrDefault( m_userParams, GEOPROCBASE_BOUNDS_OPACITY, 0.1f );

		// store the parent object name (in case sub-procedurals are re-naming it)
		m_userParams->setEntry( "parentObjectName", renderManObjectName() );
	}
	catch( napalm::NapalmError err )
	{
		DRD_LOG_DEBUG( L, "caught NapalmError" );
		throw NapalmDelightError( err.what() );
	}
	catch( ... )
	{
		throw NapalmDelightError( "uncaught exception in parse" );
	}

	// store commonly used matrices
	RtMatrix mtxA;
	Imath::M44f mtxB;

	RxTransform( "world","object",0.0f, mtxA );
	convert( mtxA, mtxB );
	setAttrDefault( m_userParams, "world2object", mtxB );

	RxTransform( "object","world",0.0f, mtxA );
	convert( mtxA, mtxB );
	setAttrDefault( m_userParams, "object2world", mtxB );

	setUpMotionBlurData();

	// set up a named coordinate system for the emitting surface
	// this could be useful when archives are emitted and "surf_P" etc needs
	// to be transformed into another space
	RiCoordinateSystem( "surf" );
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::renderBound( RtBound bound )
{
	RtColor opacity;
	opacity[0] = opacity[1] = opacity[2] = getAttr<float>( m_userParams, GEOPROCBASE_BOUNDS_OPACITY );

	RiAttributeBegin();
	RiOpacity( opacity );
	delight_box_helper::emitBox( bound );
	RiAttributeEnd();
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::optionallyRenderPrimitiveBound( RtBound bound, bool force )
{
	// early exit
	if( !force && 0 == getAttr<int>( m_userParams, GEOPROCBASE_RENDER_PRIMITIVE_BOUNDS ) )
		return;

	renderBound( bound );
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::optionallyRenderFaceBound( RtBound bound, bool force )
{
	// early exit
	if( !force && 0 == getAttr<int>( m_userParams, GEOPROCBASE_RENDER_FACE_BOUNDS ) )
		return;

	renderBound( bound );
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::setUpMotionBlurData()
{
	assert( m_timeSamples.size() == 0 );

	// see how many motion samples the user wants
	int motionSamples = getAttr< int > ( m_userParams, "motionSamples" );

	float shutterOffset = 0.0f;
	if ( getRxOption( "shutter:offset", shutterOffset ) )
	{
		DRD_LOG_DEBUG( L, "detected shutter offset: " << shutterOffset );
	}

	std::vector< float > shutterVals( 2, 0.0f );

	if (	motionSamples > 1                         // user has requested motion blur on this procedural
			&& GetRxOption( "Shutter", shutterVals )  // there's a 'Shutter' call in the rib
			&& shutterVals[0] != shutterVals[1]       // shutter open time != shutter close time
	)
	{
		assert( shutterVals.size() == 2 );
		DRD_LOG_DEBUG( L, "motion blur enabled" );
		for ( int motionSample = 0 ; motionSample < motionSamples ; ++motionSample )
		{
			float a = float( motionSample ) / ( motionSamples - 1 );
			float time = ( 1.0f - a ) * shutterVals[ 0 ] + a * shutterVals[ 1 ];
			m_timeSamples.push_back( time );
			m_offsetTimeSamples.push_back( time - shutterOffset );
			DRD_LOG_DEBUG( L, "time sample[" << motionSample << "]: " << m_timeSamples.back() );
			DRD_LOG_DEBUG( L, "offset time sample[" << motionSample << "]: " << m_offsetTimeSamples.back() );
		}
	}
	else
	{
		DRD_LOG_DEBUG( L, "motion blur disabled" );
		m_timeSamples.push_back( 0 );
		m_offsetTimeSamples.push_back( 0 );
	}

	DRD_LOG_DEBUG( L, "motion samples: " << m_timeSamples.size() );
	assert( m_timeSamples.size() != 0 );
	assert( m_offsetTimeSamples.size() == m_timeSamples.size() );
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::emitMotionBegin() const
{
	assert( getOutputTimeSamples().size() != 0 );
	if( getOutputTimeSamples().size() == 1 ) return;
	RiMotionBeginV( getOutputTimeSamples().size(), (RtFloat*) &getOutputTimeSamples()[ 0 ] );
}

//-------------------------------------------------------------------------------------------------
void GeoProcBase::emitMotionEnd() const
{
	assert( getOutputTimeSamples().size() != 0 );
	if( getOutputTimeSamples().size() == 1 ) return;
	RiMotionEnd();
}

//-------------------------------------------------------------------------------------------------
const std::vector<float>& GeoProcBase::getGxTimeSamples() const
{
	assert( m_timeSamples.size() != 0 );
	int shutterOffsetMode = getAttr< int > ( m_userParams, "shutterOffsetMode" );

	switch( shutterOffsetMode ){
		case 0:
			return m_offsetTimeSamples;
			break;
		case 1:
			return m_timeSamples;
			break;
		default:
			throw NapalmDelightError( "invalid value for shutterOffsetMode" );
	}
}

//-------------------------------------------------------------------------------------------------
const std::vector<float>& GeoProcBase::getOutputTimeSamples() const
{
	assert( m_offsetTimeSamples.size() != 0 );
	return m_offsetTimeSamples;
}
