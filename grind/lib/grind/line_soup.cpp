/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: line_soup.cpp 102193 2011-09-12 04:02:29Z luke.emrose $"
 */

//-------------------------------------------------------------------------------------------------
#include <drdDebug/log.h>
DRD_MKLOGGER(L,"drd.grind.LineSoup");
#define __class__ "LineSoup"

#include "grind/log.h"

#include "line_soup.h"
#include "bbox.h"
#include "host_vector.h"
#include "random.h"
#include "device_vector_algorithms.h"

#include "utils.h"

//#include "fur/fur_set.h"
#include <bee/gl/ScopeHelpers.h>
#include "pthread.h"

#include <boost/foreach.hpp>

#include <GL/gl.h>

#include <ri.h>

#include <list>

//! for a rainy day
#ifdef USE_GRIND_PRIM_VAR_SHADER
	#include <vector>

	#include <drdDebug/Timer.h>

	// for sampling a shader
	#include <napalmDelight/SampleSx.h>
	// for getting attributes
	#include <delightUtils/RxWrappers.h>

	#include <napalm/core/Table.h>
	#include <napalm/core/TypedBuffer.h>
	#include <napalm/core/io.h>

	#include <napalmDelight/render.h>
	#include <napalmDelight/random_helpers.h>
	#include <napalmDelight/GeoProcBase.h>
	#include <napalmDelight/SampleSx.h>
	#include <napalmDelight/attr_helpers.h>
	#include <napalmDelight/buffer_helpers.h>
	#include <napalmDelight/type_conversion.h>

	namespace n = napalm;
	using namespace napalm_delight;

#endif

using namespace grind;
using namespace drd;

//-------------------------------------------------------------------------------------------------
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

//-------------------------------------------------------------------------------------------------
extern "C"
void stupidAssignDevice( unsigned int n, float vx, float vy, float vz, float* keys, unsigned int* vals );

extern "C"
void sortIndicesDevice( unsigned int n, float* keys, unsigned int* vals );

extern "C"
void sortLineSoupIndicesDevice( unsigned int i_LineSegCount,
								float vx, float vy, float vz,
								const Imath::V3f* p,
								bool i_FurIsQuads,
								unsigned int i_IndexMidPt,
								float* keys,
								unsigned int* vals,
								unsigned int* indices );

extern "C"
void buildQuadWidthDevice(	unsigned int i_CurveCount,
							unsigned int i_CvCount,
							float vx, float vy, float vz,
							const float* width,
							Imath::V3f* p,
							Imath::V3f* n );

extern "C"
void buildQuadUVWDevice(	unsigned int i_CurveCount,
							unsigned int i_CvCount,
							Imath::V4f* o_UVW );

//-------------------------------------------------------------------------------------------------
LineSoup::LineSoup()
: m_CurveCount( 0 )
, m_CvCount( 0 )
, m_P( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
, m_N( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
, m_UVW( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
, m_IndicesGL( GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
, m_ConstantWidth( 0.001 )
, m_DirtyP( true )
, m_GeoType( LINESOUP_GL_LINES )
, m_AlignMode( LINESOUP_CAMERA_FACING )
, m_WidthMode( LINESOUP_VARYING_WIDTH )
, m_InterpolationMode( LINESOUP_CATMULL_ROM )
, m_Initialized( false )
, m_Lod( 1.0f )
, m_DisplayNormals( false )
{
	DRD_LOG_DEBUG( L, "constructing a linesoup from thread: " << ( unsigned long ) pthread_self() );
//	LOGFN0();

	m_UserParamNames.push_back( LINESOUP_PARAM_GEO_TYPE );
	m_UserParamNames.push_back( LINESOUP_PARAM_ALIGN_MODE );
	m_UserParamNames.push_back( LINESOUP_PARAM_INTERPOLATION_MODE );

}

//-------------------------------------------------------------------------------------------------
LineSoup::~LineSoup()
{
	DRD_LOG_DEBUG( L, "destroying a linesoup" );
}

void LineSoup::clear()
{
	m_CurveCount = 0;
	m_CvCount = 0;
	m_P.clear();
	m_N.clear();
	m_UVW.clear();
	m_Colour.clear();
	m_LodRand.clear();
	m_Width.clear();
	m_IndicesGL.clear();
	m_PrimVars = HostVectorSet();
	m_PrimVarTypes.clear();
	m_SortKeys.clear();
	m_SortValues.clear();
}

//-------------------------------------------------------------------------------------------------
void LineSoup::resize
(
	unsigned int i_CurveCount
,	unsigned int i_CvCount
)
{
	m_CurveCount = i_CurveCount;
	m_CvCount = i_CvCount;

	unsigned int n_elements = 0;

	if( ContextInfo::instance().hasOpenGL() )
	{
		// for GL we have two rows of verts for each curve when rendering quads
		switch ( m_GeoType )
		{
			case LINESOUP_GL_LINES:
			{
				n_elements = m_CurveCount * m_CvCount;
				break;
			}
			case LINESOUP_GL_QUADS:
			{
				n_elements = m_CurveCount * m_CvCount * 2;
				break;
			}
			default:
			{
				throw drd::RuntimeError( grindGetRiObjectName() + "unsupported geo type" );
			}
		}
	}
	else
	{
		// for other contexts, just a single row of verts for each curve
		n_elements = m_CurveCount * m_CvCount;
	}

	m_P.resize( n_elements );
	m_N.resize( n_elements );
	m_UVW.resize( n_elements );

	initIndices();
}

//-------------------------------------------------------------------------------------------------
void LineSoup::setParam( const std::string& param, const std::string& val )
{
	DRD_LOG_DEBUG( L, "trying to set param: " << param << " to: " << val );
	if( param == LINESOUP_PARAM_GEO_TYPE )
	{
		if( val == "lines" ) m_GeoType = LINESOUP_GL_LINES;
		else if( val == "quads" ) m_GeoType = LINESOUP_GL_QUADS;
		else throw drd::RuntimeError( grindGetRiObjectName() + "invalid value (should be 'lines' or 'quads'" );
		return;
	}
	else if( param == LINESOUP_PARAM_ALIGN_MODE )
	{
		if( val == "camera" ) m_AlignMode = LINESOUP_CAMERA_FACING;
		else if( val == "normal" ) m_AlignMode = LINESOUP_NORMAL_FACING;
		else throw drd::RuntimeError( grindGetRiObjectName() + "invalid value (should be 'camera' or 'normal'" );
		return;
	}
	else if( param == LINESOUP_PARAM_WIDTH_MODE )
	{
		if( val == "constant" ) m_WidthMode = LINESOUP_CONSTANT_WIDTH;
		else if( val == "varying" ) m_WidthMode = LINESOUP_VARYING_WIDTH;
		else throw drd::RuntimeError( grindGetRiObjectName() + "invalid value (should be 'constant' or 'varying'" );
		return;
	}
	else if( param == LINESOUP_PARAM_INTERPOLATION_MODE )
	{
		if( val == "linear" ) m_InterpolationMode = LINESOUP_LINEAR;
		else if( val == "catmull_rom" ) m_InterpolationMode = LINESOUP_CATMULL_ROM;
		else throw drd::RuntimeError( grindGetRiObjectName() + "invalid value (should be 'linear' or 'catmull_rom'" );
		return;
	}
	drd::RuntimeError( grindGetRiObjectName() + param + " is not a valid parameter" );
}

//-------------------------------------------------------------------------------------------------
std::vector<std::string> LineSoup::listParams( )
{

}

//-------------------------------------------------------------------------------------------------
DeviceVector< Imath::V3f >& LineSoup::getP()
{
	// indicate that P has been modified
	m_DirtyP = true;

	return m_P;
}

//-------------------------------------------------------------------------------------------------
void LineSoup::initIndices()
{
	m_Initialized = true;

	HostVector< unsigned int > buffer;
	// half way through the vert list
	unsigned int midpt = m_CurveCount * m_CvCount;

	switch ( m_GeoType )
	{
		case LINESOUP_GL_LINES:
		{
			buffer.reserve( m_CurveCount * ( m_CvCount + 1 ) );
			for ( unsigned int j = 0 ; j < m_CurveCount ; j++ )
			{
				for ( unsigned int i = 0 ; i < m_CvCount - 1 ; ++i )
				{
					buffer.push_back( j * m_CvCount + i );
					buffer.push_back( j * m_CvCount + i + 1 );
				}
			}
			break;
		}
		case LINESOUP_GL_QUADS:
		{
			buffer.reserve( m_CurveCount * ( m_CvCount + 1 ) * 2 );

			for ( unsigned int j = 0 ; j < m_CurveCount ; j++ )
			{
				for ( unsigned int i = 0 ; i < m_CvCount - 1 ; ++i )
				{
					buffer.push_back( j * m_CvCount + i );
					buffer.push_back( j * m_CvCount + i + 1 );
					buffer.push_back( j * m_CvCount + i + 1 + midpt );
					buffer.push_back( j * m_CvCount + i + midpt );
				}
			}
			break;
		}
		default:
		{
			throw drd::RuntimeError( grindGetRiObjectName() + "unsupported geometry type" );
		}
	}

	m_IndicesGL.setValue( buffer );
}

//-------------------------------------------------------------------------------------------------
void LineSoup::testSetup()
{
	// 3 line segments, 2 verts each
	resize( 3, 2 );

	HostVector<Imath::V3f>  p, n;

	p.push_back( Imath::V3f(0,0,0) );
	p.push_back( Imath::V3f(0,0,1) );

	p.push_back( Imath::V3f(0,0,0) );
	p.push_back( Imath::V3f(0,1,0) );

	p.push_back( Imath::V3f(0,0,0) );
	p.push_back( Imath::V3f(1,0,0) );

	m_P.setValue( p );
	m_Colour.setValue( p );

	n.push_back( Imath::V3f(0,0,1) );
	n.push_back( Imath::V3f(0,0,1) );

	n.push_back( Imath::V3f(0,1,0) );
	n.push_back( Imath::V3f(0,1,0) );

	n.push_back( Imath::V3f(1,0,0) );
	n.push_back( Imath::V3f(1,0,0) );

	m_N.setValue( n );

	m_WidthMode = LINESOUP_CONSTANT_WIDTH;
	m_ConstantWidth = 1;
}

//-------------------------------------------------------------------------------------------------
void LineSoup::finalizePN()
{
	// nothing to be done for renderman
	if( !ContextInfo::instance().hasOpenGL() ) return;

	// for opengl we'll be dealing with double the verts and expanding the width
	Imath::V3f view_pos = ContextInfo::instance().eyePos();

	switch ( m_GeoType )
	{
		case LINESOUP_GL_LINES:
			// nothing to be done for GL_LINES
			return;
		case LINESOUP_GL_QUADS:
			assert( m_P.size() == m_Width.size() * 2 );
			buildQuadWidthDevice(	m_CurveCount, m_CvCount,
			                      	view_pos.x, view_pos.y, view_pos.z,
			                      	m_Width.getDevicePtr(),
			                      	m_P.getDevicePtr(),
			                      	m_N.getDevicePtr() );
			return;
	}
}

//-------------------------------------------------------------------------------------------------
void LineSoup::finalizeUVW()
{
	// nothing to be done for renderman
	if( !ContextInfo::instance().hasOpenGL() ) return;

	switch ( m_GeoType )
	{
		case LINESOUP_GL_LINES:
			// nothing to be done for GL_LINES
			return;
		case LINESOUP_GL_QUADS:
			assert( m_UVW.size() == m_CurveCount * m_CvCount * 2 );
			buildQuadUVWDevice(	m_CurveCount,
			                   	m_CvCount,
								m_UVW.getDevicePtr() );
			return;
	}
}

//-------------------------------------------------------------------------------------------------
void LineSoup::viewSort() const
{
	unsigned int n_line_segments = ( m_GeoType == LINESOUP_GL_QUADS ) ? m_IndicesGL.size() / 4 :m_IndicesGL.size() / 2;
	Imath::V3f view_pos = ContextInfo::instance().eyePos();

	m_SortKeys.resize( n_line_segments );
	m_SortValues.resize( n_line_segments );

	unsigned int index_midpt = m_CurveCount * m_CvCount;

	sortLineSoupIndicesDevice( n_line_segments
	                         , view_pos.x, view_pos.y, view_pos.z
	                         , m_P.getDevicePtr()
	                         , m_GeoType == LINESOUP_GL_QUADS
	                         , index_midpt
	                         , m_SortKeys.getDevicePtr()
	                         , m_SortValues.getDevicePtr()
	                         , m_IndicesGL.getDevicePtr()
	                         );
}

//-------------------------------------------------------------------------------------------------
void LineSoup::dumpGL( float lod ) const
{
	bool do_n = m_N.size() > 0;
	bool do_uvw = m_UVW.size() > 0;
	bool do_col = m_Colour.size() > 0;

	viewSort();

	bee::glEnableHelper bl( GL_BLEND );
	bee::glEnableHelper dt( GL_DEPTH_TEST );
	bee::glBlendFuncHelper bf( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	bee::glLineWidthHelper lwh( m_ConstantWidth );

    SAFE_GL( glShadeModel( GL_SMOOTH ) );

	// positions
	SAFE_GL( glEnableClientState( GL_VERTEX_ARRAY ) );
	m_P.bindGL();
	SAFE_GL( glVertexPointer( 3, GL_FLOAT, 0, 0 ) );

	// indices
	m_IndicesGL.bindGL();

	// normals
	if( do_n )
	{
		SAFE_GL( glEnableClientState( GL_NORMAL_ARRAY ) );
		m_N.bindGL();
		SAFE_GL( glNormalPointer( GL_FLOAT, 0, 0 ) );
	}

	// uvws
	if( do_uvw )
	{
		SAFE_GL( glEnableClientState( GL_TEXTURE_COORD_ARRAY ) );
		m_UVW.bindGL();
		SAFE_GL( glTexCoordPointer( 4, GL_FLOAT, 0, 0 ) );
	}

	// colours
	if( do_col )
	{
		SAFE_GL( glEnableClientState( GL_COLOR_ARRAY ) );
		m_Colour.bindGL();
		SAFE_GL( glColorPointer( 3, GL_FLOAT, 0, 0 ) );
	}

	switch( m_GeoType ){
		case LINESOUP_GL_LINES:
			// do the actual draw
			SAFE_GL( glDrawElements( GL_LINES, m_IndicesGL.size(), GL_UNSIGNED_INT, BUFFER_OFFSET(0) ) );
			break;
		case LINESOUP_GL_QUADS:
			SAFE_GL( glDrawElements( GL_QUADS, m_IndicesGL.size(), GL_UNSIGNED_INT, BUFFER_OFFSET(0) ) );
			break;
		default:
			throw drd::RuntimeError( grindGetRiObjectName() + "unsupported geometry type" );
	}

	// reset state
	if( do_col )
	{
		SAFE_GL( glDisableClientState( GL_COLOR_ARRAY ) );
		m_Colour.unbindGL();
	}
	if( do_uvw )
	{
		SAFE_GL( glDisableClientState( GL_TEXTURE_COORD_ARRAY ) );
		m_UVW.unbindGL();
	}
	if( do_n   )
	{
		SAFE_GL( glDisableClientState( GL_NORMAL_ARRAY ) );
		m_N.unbindGL();
	}

	SAFE_GL( glDisableClientState( GL_VERTEX_ARRAY ) );

	// unbind GL
	m_IndicesGL.unbindGL();
	m_P.unbindGL();

	if( m_DisplayNormals ){
		glDisable( GL_BLEND );
		HostVector< Imath::V3f > P, N;
		P.setValue( m_P );
		N.setValue( m_N );
		assert( P.size() == N.size() );
		int n = P.size();
		glBegin( GL_LINES );
		Imath::V3f PN;
		for( int i = 0; i < n; ++i ){
			glVertex3f( P[i].x, P[i].y, P[i].z );
			PN = P[i] + N[i] * 0.1f;
			glVertex3f( PN.x, PN.y, PN.z );
		}
		glEnd();
	}
}


//-------------------------------------------------------------------------------------------------
//! extrapolate end verts
void buildCatmullRomP( int i_CurveCount, int i_CvCount, const DeviceVector<Imath::V3f>& i_P, HostVector<Imath::V3f>& o_Result )
{
	HostVector<Imath::V3f> p;
	i_P.getValue(p);
	int src_cv_id = 0;
	for( int curve = 0; curve < i_CurveCount; ++curve ){
		// extend tangent before curve
		o_Result.push_back( p[src_cv_id]*2 - p[src_cv_id+1] );

		for( int cv = 0; cv < i_CvCount; ++cv, ++src_cv_id ){
			// copy curve value
			o_Result.push_back( p[src_cv_id] );
		}
		// extend tangent after curve
		o_Result.push_back( p[src_cv_id-1]*2 - p[src_cv_id-2] );
	}
}


#define BUILD_NORMAL_DISPLAY 0

#if BUILD_NORMAL_DISPLAY
//-------------------------------------------------------------------------------------------------
// rib display of curve normals
void buildNormDisplayP( const HostVector< Imath::V3f >& i_P, const HostVector< Imath::V3f >& i_N, HostVector< Imath::V3f >& result )
{
	if( i_P.size() != i_N.size() ) throw drd::RuntimeError( grindGetRiObjectName() + "p and n should be the same size ???" );
	result.clear();
	for( int i = 0; i < i_P.size(); ++i ){
		result.push_back( i_P[i] );
		Imath::V3f n = i_N[i];
//		n = Imath::V3f(0,1,0);
		n *= 0.1f;
		result.push_back( i_P[i] + n );
	}
}
#endif

void addUniformFloatPrimVar
(
	const std::string& i_Name
,	const HostVectorSet& i_PrimVars
,	const std::map< std::string, std::string >& i_PrimVarTypes
,	std::list<std::string>& o_TokenNames
,	std::vector<RtToken>& o_Tokens
,	std::vector<RtPointer>& o_Params
)
{
	if( !i_PrimVars.hasFloatParam( i_Name ) ) return;
	const HostVector<float>& var = i_PrimVars.getFloatParam( i_Name );
	if( var.size() == 0 ) return;

	assert( i_PrimVarTypes.find( i_Name ) != i_PrimVarTypes.end() );

	std::string token = (*i_PrimVarTypes.find( i_Name )).second+" "+i_Name;
	o_TokenNames.push_back( token );
	o_Tokens.push_back( &o_TokenNames.back()[0] );
	o_Params.push_back( (RtPointer)&var[0] );
	DRD_LOG_INFO( L,  token << ": min=" << *(std::min_element( var.begin(), var.end() ))
									   << ", max=" << *(std::max_element( var.begin(), var.end() )));
}

void addUniformVec3fPrimVar
(
	const std::string& i_Name
,	const HostVectorSet& i_PrimVars
,	const std::map< std::string, std::string >& i_PrimVarTypes
,	std::list<std::string>& o_TokenNames
,	std::vector<RtToken>& o_Tokens
,	std::vector<RtPointer>& o_Params
)
{
	if( !i_PrimVars.hasVec3fParam( i_Name ) ) return;
	const HostVector<Imath::V3f>& var = i_PrimVars.getVec3fParam( i_Name );
	if( var.size() == 0 ) return;

	assert( i_PrimVarTypes.find( i_Name ) != i_PrimVarTypes.end() );

	std::string token = ( *i_PrimVarTypes.find( i_Name ) ).second + " " + i_Name;
	o_TokenNames.push_back( token );
	o_Tokens.push_back( &o_TokenNames.back()[ 0 ] );
	o_Params.push_back( ( RtPointer ) &var[ 0 ] );
}

//-------------------------------------------------------------------------------------------------
void LineSoup::info() const
{
	std::cout << "LineSoup:\n";
	std::cout << "  curves: " << m_CurveCount << std::endl;
	std::cout << "  cvs:    " << m_CvCount << std::endl;
	switch( m_AlignMode ){
		case LINESOUP_CAMERA_FACING: std::cout << "  camera facing\n"; break;
		case LINESOUP_NORMAL_FACING: std::cout << "  normal facing\n"; break;
		default: break;
	}

	switch( m_GeoType ){
		case LINESOUP_GL_LINES: std::cout << "  lines\n"; break;
		case LINESOUP_GL_QUADS: std::cout << "  quads\n"; break;
		default: break;
	}

	switch( m_WidthMode ){
		case LINESOUP_CONSTANT_WIDTH: std::cout << "  constant width\n"; break;
		case LINESOUP_VARYING_WIDTH: std::cout << "  varying width\n"; break;
		default: break;
	}
}

//-------------------------------------------------------------------------------------------------
void LineSoup::dumpRib( float lod ) const
{
	LOGFN1( lod );

	// ensure that we have run initIndices before we get to this point
	assert( m_Initialized );

	// early out for zero curves
	if( m_CurveCount == 0 )
		return;

	// expose the max lod for shader access
	// TODO: it's BAD to do this here, this needs to be instead done by the client code
	//       which in this case would likely be the python procedural
	//RiAttribute( "user", "float lod", ( RtPointer ) &m_Lod, RI_NULL );
	DRD_LOG_DEBUG( L, "RiAttribute float lod = " << m_Lod );

	std::list< std::string > token_names;
	std::vector< RtToken > tokens;
	std::vector< RtPointer > params;

#if 1

	HostVector< Imath::V3f > p;
	HostVector< Imath::V3f > n;
	HostVector< RtInt > nverts;
	HostVector< float > width;

	switch( m_InterpolationMode )
	{
		case LINESOUP_LINEAR:
		{
			nverts.resize( m_CurveCount, m_CvCount );
			m_P.getValue( p );
			break;
		}
		case LINESOUP_CATMULL_ROM:
		{
			nverts.resize( m_CurveCount, m_CvCount+2 );
			buildCatmullRomP( m_CurveCount, m_CvCount, m_P, p );
			break;
		}
		default:
		{
			throw drd::RuntimeError( grindGetRiObjectName() + "unsupported interpolation mode" );
		}
	}

#if 0
#if LINEAR_INTERPOLATION
	DRD_LOG_DEBUG( L, "*** LINEAR INTERPOLATION ***" );
	m_P.getValue( p );
#else
	buildRendermanP( m_CurveCount, m_CVCount, m_P, p );
#endif
#endif
	tokens.push_back( RI_P );
	params.push_back( &( p[ 0 ] ) );

	if( m_AlignMode == LINESOUP_NORMAL_FACING )
	{
		m_N.getValue( n );
		tokens.push_back( "varying normal N" );
		params.push_back( &n[0] );
	}

	{
		std::vector< std::string > float_prim_var_names;
		m_PrimVars.getFloatParamNames( float_prim_var_names );
		std::vector< std::string >::const_iterator i(float_prim_var_names.begin());
		std::vector< std::string >::const_iterator e(float_prim_var_names.end());
		for( ; i!=e; ++i )
		{
			addUniformFloatPrimVar( *i, m_PrimVars, m_PrimVarTypes, token_names, tokens, params );
		}
	}

	{
		std::vector< std::string > vec_prim_var_names;
		m_PrimVars.getVec3fParamNames( vec_prim_var_names );
		std::vector< std::string >::const_iterator i(vec_prim_var_names.begin());
		std::vector< std::string >::const_iterator e(vec_prim_var_names.end());
		for( ; i!=e; ++i ){
			addUniformVec3fPrimVar( *i, m_PrimVars, m_PrimVarTypes, token_names, tokens, params );
		}
	}


	switch( m_WidthMode )
	{
		case LINESOUP_VARYING_WIDTH:
			m_Width.getValue( width );
			tokens.push_back( "width" );
			params.push_back( &width[0] );
			break;
		case LINESOUP_CONSTANT_WIDTH:
			tokens.push_back( "constantwidth" );
			params.push_back(  const_cast< float* >( &m_ConstantWidth ) );
			break;
		default:
			throw drd::RuntimeError( grindGetRiObjectName() + "unsupported width mode" );
	}

//! maybe for a rainy day....
#ifdef USE_GRIND_PRIM_VAR_SHADER
	//----------------------------------------------------------------------------
	//----------------------------------------------------------------------------
	//----------------------------------------------------------------------------
	//----------------------------------------------------------------------------

	// data storage
	std::vector< float > root_occl;
	std::vector< float > tip_occl;

	// load in the shader
	std::string shaderFileName;

	//----------------------------------------------------------------------------
	// obtain the units being used to specify time
	bool o_shutter_in_seconds = true;
	std::vector< std::string > string_data( 1 );
	if( drd::GetRxAttribute( "user:grindPrimVarShader", string_data ) )
	{
		// find the root_normal, root_p and tip_p variables
		int root_normal_id = -1;
		int root_p_id = -1;
		int tip_p_id = -1;
		for( std::vector< RtToken >::const_iterator i = tokens.begin(); i != tokens.end(); ++i )
		{
			if( strcmp( *i, "uniform normal root_normal" ) == 0 )
			{
				root_normal_id = i - tokens.begin();
			}
			else if( strcmp( *i, "uniform point root_p" ) == 0 )
			{
				root_p_id = i - tokens.begin();
			}
			else if( strcmp( *i, "uniform point tip_p" ) == 0 )
			{
				tip_p_id = i - tokens.begin();
			}
		}

		// if we have all the data we need, then continue
		if( ( root_normal_id != -1 ) && ( root_p_id != -1 ) && ( tip_p_id != -1 ) )
		{
			// get the shader file name
			shaderFileName = string_data[ 0 ];

			// result curves data
			size_t numCurves = nverts.size();
			size_t numCurveVerts = nverts[ 0 ];

			// set up the shader parameters
			n::object_table_ptr shaderParams( new n::ObjectTable() );
			shaderParams->setEntry( "name", "furOcclusionKernel2" );

			// the bakefile parameter
			n::object_table_ptr bakefile( new n::ObjectTable() );
			bakefile->setEntry( 0, "body.ptc" );

			// the params parameter
			n::object_table_ptr paramsTable( new n::ObjectTable() );
			paramsTable->setEntry( "bakefile", bakefile );
			shaderParams->setEntry( "params", paramsTable );

			// set up the sample evaluator
			// create a copy of the parent context so that we can change it
			drd::SxContext parentContext = drd::SxContext( new SxContextWrapper( ::SxCreateContext( SX_RI_PARENT_CONTEXT ) ) );
			std::vector< int > nthreads( 1, 8 );
			SxSetOption( parentContext, "render:nthreads", nthreads );
			SampleSx primVarShaderSx( shaderParams, *( parentContext->get_data() ) );

			//----------------------------------------------------------------------------
			// root_p, tip_p
			//----------------------------------------------------------------------------
			// populate the input variables
			n::object_table_ptr callVars( new n::ObjectTable() );

			// populate position, which is the root and tip values alternating
			n::V3fBufferPtr P( new n::V3fBuffer( numCurves * 2 ) );
			{
				n::V3fBuffer::w_type P_access = P->w();
				n::V3fBuffer::w_type::iterator P_iter = P_access.begin();
				for( size_t i = 0; i < numCurves; ++i )
				{
					// copy the normal twice, once for the root and once for the tip
					memcpy( &( ( *P_iter++ )[ 0 ] ), &( ( reinterpret_cast< float* >( params[ root_p_id ] ) )[ i * 3 ] ), sizeof( float ) * 3 );
					memcpy( &( ( *P_iter++ )[ 0 ] ), &( ( reinterpret_cast< float* >( params[ tip_p_id ] ) )[ i * 3 ] ), sizeof( float ) * 3 );
				}
			}
			callVars->setEntry( "P", P );
			//----------------------------------------------------------------------------
			// root_p, tip_p
			//----------------------------------------------------------------------------

			//----------------------------------------------------------------------------
			// root_normal
			//----------------------------------------------------------------------------
			// populate the normal, which is the root_normal twice, once for root and tip points
			n::V3fBufferPtr N( new n::V3fBuffer( numCurves * 2 ) );
			{
				n::V3fBuffer::w_type N_access = N->w();
				n::V3fBuffer::w_type::iterator N_iter = N_access.begin();
				for( size_t i = 0; i < numCurves; ++i )
				{
					// copy the normal twice, once for the root and once for the tip
					memcpy( &( ( *N_iter++ )[ 0 ] ), &( ( reinterpret_cast< float* >( params[ root_normal_id ] ) )[ i * 3 ] ), sizeof( float ) * 3 );
					memcpy( &( ( *N_iter++ )[ 0 ] ), &( ( reinterpret_cast< float* >( params[ root_normal_id ] ) )[ i * 3 ] ), sizeof( float ) * 3 );
				}
			}
			callVars->setEntry( "N", N );
			//----------------------------------------------------------------------------
			// root_normal
			//----------------------------------------------------------------------------

			std::cerr << "Using Sx to shade: " << p.size() << " points...." << std::endl;

			// set up the output variables
			n::ObjectTable aovs;
			aovs.setEntry( 0, "occl" );

			// result table
			n::object_table_ptr result( new n::ObjectTable() );

			std::cerr << "parmsTable" << std::endl;
			paramsTable->Object::dump( std::cerr );

			std::cerr << "callVars" << std::endl;
			callVars->Object::dump( std::cerr );

			std::cerr << "aovs" << std::endl;
			aovs.Object::dump( std::cerr );

			{
				// start timer
				drd::Timer sxTimer;

				// compute
				primVarShaderSx.sample( *callVars, aovs, *result );

				// end timer
				std::string temp;
				sxTimer.getDescription( temp );
				std::cerr << temp << std::endl;
			}

			std::cerr << "result" << std::endl;
			result->Object::dump( std::cerr );

			n::FloatBufferPtr occl;
			if( result->getEntry( "occl", occl ) )
			{
				// reserve some space for the results
				root_occl.resize( numCurves, 0 );
				tip_occl.resize( numCurves, 1 );

				// get the result
				n::FloatBuffer::r_type occl_access = occl->r();
				n::FloatBuffer::r_type::iterator occl_iter = occl_access.begin();
				for( size_t i = 0; i < numCurves; ++i )
				{
					// get the root and tip results....

					// get the root
					root_occl[ i ] = *occl_iter++;

					// get tip
					tip_occl[ i ] = *occl_iter++;
				}

				// add these vars to the
				tokens.push_back( "uniform float root_occl" );
				params.push_back( &root_occl[ 0 ] );

				tokens.push_back( "uniform float tip_occl" );
				params.push_back( &tip_occl[ 0 ] );
			}
			else
			{
				DRD_LOG_ERROR( L, "unable to find Ci AOV in: " << shaderFileName );
			}
		}
		else
		{
			DRD_LOG_ERROR( L, "unable to find root_normal, root_p and tip_p primvars on the current fur object...." );
		}
	}

	//----------------------------------------------------------------------------
	//----------------------------------------------------------------------------
	//----------------------------------------------------------------------------
	//----------------------------------------------------------------------------
#endif

//!
#if 0
		nverts.clear();
	#if LINEAR_INTERPOLATION
		nverts.resize( m_CurveCount, m_CVCount );
	#else
		nverts.resize( m_CurveCount, m_CVCount+2 );
	#endif
#endif

	DRD_LOG_DEBUG( L, "Emitting " << nverts.size() << " RiCurves (vars: " );
	BOOST_FOREACH( const RtToken& t, tokens )
	{
		DRD_LOG_DEBUG( L, "\t" << t << ", " );
	}
	DRD_LOG_DEBUG( L, ")" );

#if 0
	DRD_LOG_DEBUG( L, "p:" );			p.dump();
	DRD_LOG_DEBUG( L, "n:" );			n.dump();
	DRD_LOG_DEBUG( L, "width:" );		width.dump();
	DRD_LOG_DEBUG( L, "m_HairId:" );	m_HairId.dump();
	DRD_LOG_DEBUG( L, "m_LodRand:" );	m_LodRand.dump();
	DRD_LOG_DEBUG( L, "m_RootS:" );		m_RootS.dump();
	DRD_LOG_DEBUG( L, "m_RootT:" );		m_RootT.dump();
#endif

	switch( m_InterpolationMode )
	{

		case LINESOUP_LINEAR:
			RiCurvesV( RI_LINEAR, nverts.size(), &nverts[0], RI_NONPERIODIC, tokens.size(), &tokens[0], &params[0] );
			break;

		case LINESOUP_CATMULL_ROM:
			// TODO: this needs to be set in the calling code, not here....
			//RiBasis( RiCatmullRomBasis, 1, RiCatmullRomBasis, 1 );
			RiCurvesV( RI_CUBIC, nverts.size(), &nverts[0], RI_NONPERIODIC, tokens.size(), &tokens[0], &params[0] );
			break;

		default:
			throw drd::RuntimeError( grindGetRiObjectName() + "unsupported interpolation mode" );
	}
#if 0
#if LINEAR_INTERPOLATION
	RiCurvesV( RI_LINEAR, nverts.size(), &nverts[0], RI_NONPERIODIC, tokens.size(), &tokens[0], &params[0] );
#else
	RiBasis( RiCatmullRomBasis, 1, RiCatmullRomBasis, 1 );
	RiCurvesV( RI_CUBIC, nverts.size(), &nverts[0], RI_NONPERIODIC, tokens.size(), &tokens[0], &params[0] );
#endif
#endif


#if BUILD_NORMAL_DISPLAY
	// make sure p and n are correct
	m_P.getValue( p );
	m_N.getValue( n );

	HostVector< Imath::V3f > norm_display_p;
	buildNormDisplayP( p, n, norm_display_p );

	tokens.clear();
	params.clear();
	tokens.push_back( RI_P );
	params.push_back( &norm_display_p[0] );

	nverts.clear();
	nverts.resize( norm_display_p.size() / 2, 2 );

	tokens.push_back( "constantwidth" );
	params.push_back(  &constantwidth );

	RiCurvesV( RI_LINEAR, nverts.size(), &nverts[0], RI_NONPERIODIC, tokens.size(), &tokens[0], &params[0] );
#endif


#else

	/*
	 *
	 * JUST TESTING
	 *
	 */

	HostVector<Imath::V3f> p;
	HostVector<Imath::V3f> n;

	int n_curves = 6;
	int n_cvs = 4;

	for( int i = 0; i < n_curves * (n_cvs+2); ++i ){
		p.push_back( Imath::V3f( 0,float(i)/n_curves * 0.8 + 2.0f, i % 2 ) );
	}
	tokens.push_back( RI_P );
	params.push_back( &p[0] );

#if 1
	for( int c = 0; c < n_curves; ++c ){
		for( int cv = 0; cv < n_cvs; ++cv ){
			// alternate for each curve to check
			if( c %2 == 0 ){
				n.push_back( Imath::V3f( 0, 0, 1 ) );
			} else {
				n.push_back( Imath::V3f( 1, 0, 0 ) );
			}
		}
	}
	tokens.push_back( RI_N );
	params.push_back( &n[0] );
#endif



	HostVector<RtInt> nverts;
	nverts.resize( n_curves, n_cvs+2 );

#if 1
	HostVector<float> width;
	for( int c = 0; c < n_curves; ++c ){
		for( int cv = 0; cv < n_cvs; ++cv ){
			width.push_back( 0.1 * (float(cv) / n_cvs-1) );
		}
	}
	tokens.push_back( "width" );
	params.push_back( &width[0] );

#else
	tokens.push_back( "constantwidth" );
	params.push_back(  &constantwidth );
#endif

#if 1
	RiBasis( RiCatmullRomBasis, 1, RiCatmullRomBasis, 1 );
	RiCurvesV( RI_CUBIC, nverts.size(), &nverts[0], RI_NONPERIODIC, tokens.size(), &tokens[0], &params[0] );
#else
	RiCurvesV( RI_LINEAR, nverts.size(), &nverts[0], RI_NONPERIODIC, tokens.size(), &tokens[0], &params[0] );
#endif

	tokens.clear();
	params.clear();
	tokens.push_back( RI_P );
	params.push_back( &p[0] );

	tokens.push_back( "constantwidth" );
	params.push_back(  &constantwidth );

	RiPointsV( p.size(), tokens.size(), &tokens[0], &params[0] );
#endif
}


//-------------------------------------------------------------------------------------------------
BBox LineSoup::getBounds() const
{
	LOGFN0();

	// get bounds of verts
	BBox result = grind::getBounds( m_P );

	// pad by the maximum width value
	result.pad( grind::getMaxValue( m_Width ) );

	// for now, recalculate on each request (could be cached with dirty flag etc)
	return result;
}

//-------------------------------------------------------------------------------------------------
LineSoup::GeoType LineSoup::getActualGeoType() const
{
	// will only have quad data if the user has it set, and we're in OpenGL
	return ContextInfo::instance().hasOpenGL() && m_GeoType == LINESOUP_GL_QUADS ? LINESOUP_GL_QUADS : LINESOUP_GL_LINES;
}

//-------------------------------------------------------------------------------------------------
//! return the alignment mode used by this instance
LineSoup::AlignMode LineSoup::getAlignMode() const
{
	return m_AlignMode;
}

//-------------------------------------------------------------------------------------------------
void LineSoup::getData( const std::string& name, std::vector< Imath::V3f >& result ) const
{
	if( name == "P" ){
		m_P.getValue( result );
		return;
	}
	if( name == "N" ){
		m_N.getValue( result );
		return;
	}
	throw drd::RuntimeError( grindGetRiObjectName() + std::string( "trying to get vector data named '" ) + name + "'" );
}

//-------------------------------------------------------------------------------------------------
void LineSoup::setData( const std::string& name, const std::vector< Imath::V3f >& src )
{
	if( name == "P" ){
		m_P.setValue( src );
		return;
	}
	if( name == "N" ){
		m_N.setValue( src );
		return;
	}
	throw drd::RuntimeError( grindGetRiObjectName() + std::string( "trying to set vector data named '" ) + name + "'" );
}


/***
    Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)

    This file is part of anim-studio-tools.

    anim-studio-tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    anim-studio-tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.
***/
