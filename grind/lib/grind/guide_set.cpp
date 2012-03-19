/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: guide_set.cpp 99443 2011-08-30 01:14:50Z hugh.rayner $"
 */

//-------------------------------------------------------------------------------------------------

#include <drdDebug/log.h>
DRD_MKLOGGER( L, "drd.grind.GuideSet" );

#include "grind/utils.h"

#include "grind/guide_set.h"
#include "grind/host_vector.h"
#include "grind/log.h"
#include "grind/mesh.h"
#include "grind/random.h"
#include "grind/device_vector_algorithms.h"

#include <GL/glew.h>

// bee includes
#include <bee/io/Loader.h>
#include <bee/io/MeshLoader.h>

#include <stdexcept>
#include <ri.h>


//-------------------------------------------------------------------------------------------------
using namespace grind;
using namespace drd;
using namespace std;

//-------------------------------------------------------------------------------------------------
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#define RestartIndex 0xffffffff

//-------------------------------------------------------------------------------------------------
extern "C"
void modUsingCuda( void* p, unsigned int n );

extern "C"
void surfaceNormalGroomDevice( 	unsigned int n_curves,
								unsigned int n_cvs,
								Imath::V3f* mesh_p,
								Imath::V3f* mesh_n,
								const float* span_length,
								Imath::V3f* p,
								Imath::V3f* prev_p );

extern "C"
void tangentSpaceUpdateDevice( 	unsigned int n_curves,
								unsigned int n_cvs,
								const Imath::V3f* guide_p_tangent,
								const Imath::V3f* guide_across_tangent,
								const Imath::V3f* mesh_p,
								const Imath::V3f* mesh_n,
								const Imath::V3f* mesh_t,
								Imath::V3f* guide_p,
								Imath::V3f* guide_across );

extern "C"
void guideBuildAcrossDisplayDevice( unsigned int n_curves,
                                    unsigned int n_cvs,
                                    const Imath::V3f* guide_p,
                                    const Imath::V3f* guide_across,
                                    Imath::V3f* guide_across_display );

extern "C"
void calcCurveLengthDevice( unsigned int i_CurveCount,
                            unsigned int i_CvCount,
                            const Imath::V3f* i_GuideP,
                            float* o_GuideLength );

//-------------------------------------------------------------------------------------------------
GuideSet::GuideSet()
: m_Indices( GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
, m_CurveCount( 0 )
, m_CVCount( 0 )
, m_P( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
, m_PTangent( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
, m_AcrossDisplay( GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW )
{
}

//-------------------------------------------------------------------------------------------------
void GuideSet::dumpGL( float lod ) const
{
	if ( m_P.size() == 0 ) return;

	SAFE_GL( glPrimitiveRestartIndexNV( RestartIndex ) );
	SAFE_GL( glEnableClientState( GL_PRIMITIVE_RESTART_NV) );

	m_Indices.bindGL();
	m_P.bindGL();

	SAFE_GL( glEnableClientState( GL_VERTEX_ARRAY) );
	SAFE_GL( glVertexPointer( 3, GL_FLOAT, 0, 0 ) );

	// segments
	SAFE_GL( glColor4f( 1.0, 0.0, 0.0, 1.0 ) );
	SAFE_GL( glLineWidth( 1 ) );
	SAFE_GL( glDrawElements( GL_LINE_STRIP, m_CurveCount * ( m_CVCount + 1 ), GL_UNSIGNED_INT, BUFFER_OFFSET(0) ) );

	// cvs
	SAFE_GL( glColor4f( 1.0, 1.0, 1.0, 1.0 ) );
	SAFE_GL( glPointSize( 1 ) );
	SAFE_GL( glDrawArrays( GL_POINTS, 0, m_CurveCount * m_CVCount ) );

	SAFE_GL( glDisableClientState( GL_PRIMITIVE_RESTART_NV ) );
	SAFE_GL( glDisableClientState( GL_VERTEX_ARRAY ) );

	m_Indices.unbindGL();
	m_P.unbindGL();

	// across vector display
	if( m_Across.size() != 0 ){
		m_AcrossDisplay.resize( m_Across.size() * 2 );
		guideBuildAcrossDisplayDevice( m_CurveCount, m_CVCount, m_P.getDevicePtr(), m_Across.getDevicePtr(), m_AcrossDisplay.getDevicePtr() );
		m_AcrossDisplay.bindGL();
		SAFE_GL( glEnableClientState( GL_VERTEX_ARRAY) );
		SAFE_GL( glVertexPointer( 3, GL_FLOAT, 0, 0 ) );
		SAFE_GL( glColor4f( 0.0, 1.0, 0.0, 1.0 ) );
		SAFE_GL( glDrawArrays( GL_LINES,  0, m_Across.size() * 2 ) );
		SAFE_GL( glDisableClientState( GL_VERTEX_ARRAY ) );
	}

}

//!-------------------------------------------------------------------------------------------------
//! this attempts to keep consistent with dumpGL
void GuideSet::dumpRib( float lod ) const
{
	if ( m_P.size() == 0 ) return;

	// token storage
	std::vector< RtToken > tokens;
	std::vector< RtPointer > params;

	// data storage
	HostVector< Imath::V3f > p;
	m_P.getValue( p );

	// collect "P"
	tokens.push_back( RI_P );
	params.push_back( &( p[ 0 ] ) );

	// red colour to emit with
	RtColor red = { 1.0f, 0.0f, 0.0f };
	tokens.push_back( "constant color Cs" );
	params.push_back( red );

	RtColor transp = { 0.1f, 0.1f, 0.1f };
	tokens.push_back( "constant color Os" );
	params.push_back( transp );

	// point width
	float width = 0.0025f;
	tokens.push_back( RI_CONSTANTWIDTH );
	params.push_back( &width );

	// for each curve, store its number of CV's
	std::vector< int > nverts;
	nverts.resize( m_CurveCount, m_CVCount );

	// emit the curves themselves
	RiCurvesV( RI_LINEAR, nverts.size(), &nverts[ 0 ], RI_NONPERIODIC, tokens.size(), &tokens[ 0 ], &params[ 0 ] );

	//int totalNumVerts = m_CurveCount * m_CVCount;
	//for( int i = 0; i < totalNumVerts; ++i )
	//{
	//	std::cout << p[ i ] << std::endl;
	//}
}

//-------------------------------------------------------------------------------------------------
void GuideSet::randomize( 	unsigned int n_curves,
							unsigned int n_cvs )
{
	m_CurveCount = n_curves;
	m_CVCount = n_cvs;
	unsigned int n_pts = m_CurveCount * m_CVCount;
	HostVector< Imath::V3f > p;
	p.resize( n_pts );

	for ( int i = 0 ; i < n_pts ; ++i )
	{
		p[ i ].x = nRand();
		p[ i ].y = nRand();
		p[ i ].z = nRand();
	}

	m_P.setValue( p );
	setupIndices();
}

//-------------------------------------------------------------------------------------------------
void GuideSet::setupIndices()
{
	HostVector< unsigned int > indices;
	indices.resize( m_CurveCount * ( m_CVCount + 1 ) );

	HostVector< unsigned int >::iterator iter( indices.begin() );

	for ( unsigned int j = 0 ; j < m_CurveCount ; j++ )
	{
		for ( unsigned int i = 0 ; i < m_CVCount ; ++i )
		{
			*iter++ = j * m_CVCount + i;
		}
		*iter++ = RestartIndex;
	}

	m_Indices.setValue( indices );
}

//-------------------------------------------------------------------------------------------------
void GuideSet::resize( 	unsigned int n_curves,
						unsigned int n_cvs )
{
	m_CurveCount = n_curves;
	m_CVCount = n_cvs;

	int total_cvs = m_CurveCount * m_CVCount;

	m_P.resize( total_cvs );
	m_PrevP.resize( total_cvs );
	m_SpanLength.resize( total_cvs );

	setupIndices();
}

//-------------------------------------------------------------------------------------------------
void GuideSet::init( 	unsigned int n_curves,
						int n_cvs,
						float guide_length )
{
	resize( n_curves, n_cvs );

	HostVector< GuideParams > params( m_CurveCount );

	for ( int g = 0 ; g < m_CurveCount ; ++g )
	{
		params[ g ].stiffness_root = 1.0f * mRand( 0.0f );
		params[ g ].stiffness_tip = 0.1f * mRand( 0.25f );
		params[ g ].stiffness_gamma = 0.2f * mRand( 0.25f );
	}

	m_Params.setValue( params );

	HostVector< float > sl( m_CurveCount, guide_length / ( n_cvs - 1 ) );
	m_SpanLength.setValue( sl );
}

//-------------------------------------------------------------------------------------------------
void GuideSet::surfaceNormalGroom( const DeviceMesh& mesh )
{
	if ( m_CurveCount != mesh.nVerts() )
	{
		throw drd::RuntimeError( grindGetRiObjectName() + "guides need to be initialized before grooming" );
	}

	surfaceNormalGroomDevice( m_CurveCount, m_CVCount, mesh.getP().getDevicePtr(), mesh.getN().getDevicePtr(), m_SpanLength.getDevicePtr(), m_P.getDevicePtr(),
			m_PrevP.getDevicePtr() );

	HostVector< Imath::V3f> verts;
	m_P.getValue( verts );
	snapshotIntoTangentSpace( mesh, verts );
	calcGuideLength();
}

//-------------------------------------------------------------------------------------------------
void GuideSet::calcGuideLength() const
{
	m_GuideLength.resize( m_CurveCount );
	assert( m_P.size() == m_CurveCount * m_CVCount );
	calcCurveLengthDevice( m_CurveCount, m_CVCount, m_P.getDevicePtr(), m_GuideLength.getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------

static boost::shared_ptr<const bee::MeshLoader> getMeshLoader(const std::string &a_FilePath, float a_Frame)
{
	// serialize access to this cache
	static boost::mutex mx;
	boost::mutex::scoped_lock sl(mx);
	static std::map<std::pair<std::string,float>, boost::weak_ptr<const bee::MeshLoader> > mesh_loader_cache;

	std::pair<std::string,float> key(a_FilePath,a_Frame);

	boost::shared_ptr< const bee::MeshLoader > result;
	std::map<std::pair<std::string,float>, boost::weak_ptr<const bee::MeshLoader> >::const_iterator i = mesh_loader_cache.find(key);
	if(i != mesh_loader_cache.end())
	{
		result = (*i).second.lock();
	}

	if( !result ) // have to load, or reload, the mesh
	{
		boost::shared_ptr< bee::MeshLoader > m_Loader = bee::Loader::CreateMeshLoader( a_FilePath );
		m_Loader->open();
		m_Loader->reportStats();
		if ( !m_Loader->setFrame( a_Frame ) )
		{
			DRD_LOG_WARN( L, "Was NOT able to set frame to: " << a_Frame );
			m_Loader.reset();
		}
		mesh_loader_cache[key] = m_Loader;
		result = m_Loader;
	}
	else
	{
		;
	}
	return result;
}

bool GuideSet::setFrame( float a_Frame, const DeviceMesh& a_Mesh )
{
	m_Loader = getMeshLoader(m_FilePath,a_Frame);
	if ( !m_Loader )
	{
		DRD_LOG_WARN( L, "no mesh: " << m_FilePath );
		return false;
	}

    try
    {
        const std::vector< Imath::V3f > & vertexVector = m_Loader->getVertexVector();
		HostVector< Imath::V3f > obj_verts;
        obj_verts.assign( vertexVector.begin(), vertexVector.end() );

        DRD_LOG_DEBUG( L, "Sizes: " << obj_verts.size() << ":" << a_Mesh.nVerts() );

		if ( obj_verts.size() == 0 || a_Mesh.nVerts() == 0 )
		{
			throw drd::RuntimeError( grindGetRiObjectName() + "Not enough data." );
            return false;
		}

		if( obj_verts.size() % a_Mesh.nVerts() != 0 )
		{
			std::ostringstream oss;
			oss << "guide vert count (" << obj_verts.size() << ") doesn't match mesh vert count (" << a_Mesh.nVerts() << ")";
			throw drd::RuntimeError( grindGetRiObjectName() + oss.str() );
            return false;
		}
		else
		{
			DRD_LOG_INFO( L, "guide cv count: " << obj_verts.size() / a_Mesh.nVerts() );
		}

		m_CurveCount = a_Mesh.nVerts();
		m_CVCount = obj_verts.size() / m_CurveCount;

		snapshotIntoTangentSpace( a_Mesh, obj_verts );

		m_P.setValue( obj_verts );

		calcGuideLength();
		setupIndices();

	}
	catch ( const std::exception& e )
	{
		throw drd::RuntimeError( grindGetRiObjectName() + std::string( "GuideSet: error setFrame() : " ) + e.what() );
        return false;
	}
	return true;
}

//-------------------------------------------------------------------------------------------------
void GuideSet::read( const std::string& i_FilePath, const DeviceMesh& a_Mesh )
{
	try
	{
		m_FilePath = i_FilePath;
		setFrame( 1.0f, a_Mesh );

	}
	catch ( const std::exception& e )
	{
		throw drd::RuntimeError( grindGetRiObjectName() + std::string( "GuideSet: error loading from '" ) + i_FilePath + "': " + e.what() );
	}
}

//-------------------------------------------------------------------------------------------------
void GuideSet::update( const DeviceMesh& i_Mesh )
{
	if ( m_PTangent.size() == 0 )
	{
		throw drd::RuntimeError( grindGetRiObjectName() + "can't find tangent space guides" );
	}

	tangentSpaceUpdateDevice( m_CurveCount,
	                          m_CVCount,
	                          m_PTangent.getDevicePtr(),
	                          m_AcrossTangent.getDevicePtr(),
	                          i_Mesh.getP().getDevicePtr(),
	                          i_Mesh.getN().getDevicePtr(),
	                          i_Mesh.getTangent().getDevicePtr(),
	                          m_P.getDevicePtr(),
	                          m_Across.getDevicePtr() );
}

//-------------------------------------------------------------------------------------------------
void GuideSet::snapshotIntoTangentSpace( const DeviceMesh& i_Mesh, const HostVector< Imath::V3f > & i_GuideVerts )
{
	HostVector< Imath::V3f > mesh_p;
	HostVector< Imath::V3f > mesh_normals;
	HostVector< Imath::V3f > mesh_tangents;
	HostVector< Imath::V3f > across_vectors;

	i_Mesh.getStaticP().getValue( mesh_p );
	i_Mesh.getStaticN().getValue( mesh_normals );
	i_Mesh.getStaticTangent().getValue( mesh_tangents );

	if( mesh_normals.size() == 0 ) throw drd::RuntimeError( grindGetRiObjectName() + "surface mesh has no surface normals" );
	if( mesh_tangents.size() == 0 ) throw drd::RuntimeError( grindGetRiObjectName() + "surface mesh has no tangents" );

	HostVector< Imath::V3f > result;

	// work out the bounds of the static p
	BBox bounds = grind::getBounds( i_Mesh.getStaticP() );

	const float mesh_diagonal = bounds.GetBox().size().length();

	// sanity check: max deviation allowed between mesh vert and root guide vert is 1% of diagonal
	const float root_threshold = mesh_diagonal * 0.01f;

	// host code since it only has to be done once per mesh (not per frame) and is pretty trivial
	for( int c = 0; c < m_CurveCount; ++c )
    {
		const Imath::V3f& mesh_root = mesh_p[ c ];
		const Imath::V3f& N = mesh_normals[ c ];
		const Imath::V3f& T = mesh_tangents[ c ];

		// validate inputs
		assert( fabsf( N.length() - 1.0f ) < 1e-3 ); // unit length
		assert( fabsf( T.length() - 1.0f ) < 1e-3 ); // unit length
		assert( fabsf( N.dot( T ) ) < 1e-3 ); // orthogonal

		const Imath::V3f & root = i_GuideVerts[c*m_CVCount];
		if( (root - mesh_root).length() > root_threshold )
        {
			DRD_LOG_VERBATIM( L, "mesh bounds: " << bounds.GetBox().min << " - " << bounds.GetBox().max );
			DRD_LOG_VERBATIM( L, "mesh bound transform: " << bounds.GetTransform() );
			DRD_LOG_VERBATIM( L, "root threshold (1% of bbox diagonal): " << root_threshold );
			DRD_LOG_VERBATIM( L, "Guide root vert: " << root );
			DRD_LOG_VERBATIM( L, "Mesh vert: " << mesh_root );
			DRD_LOG_ERROR( L, grindGetRiObjectName() << "guide id " << c << " doesn't match mesh. (exceeds 1% of bbox diagonal)" );

			throw drd::RuntimeError( grindGetRiObjectName() + "guide tolerence exceeded (see previous errors)" );
		}

		// binormal
		Imath::V3f BN = T.cross(N);

		for( int cv = 0; cv < m_CVCount; ++cv )
        {
			Imath::V3f delta = i_GuideVerts[ c * m_CVCount + cv ] - root;
			result.push_back( Imath::V3f( delta.dot( T ), delta.dot( N ), delta.dot( BN ) ) );
		}

		// across vector in object space (tip of hair - root of hair)
		Imath::V3f across = (( i_GuideVerts[ (c+1) * m_CVCount - 1 ] - i_GuideVerts[ c * m_CVCount ] ).cross( N )).normalized();
		// store in tangent space
		Imath::V3f across_t( across.dot( T ), across.dot( N ), across.dot( BN ) );
		across_vectors.push_back( across_t );
	}

	m_PTangent.setValue( result );
	m_P.resize( m_PTangent.size() );
	m_AcrossTangent.setValue( across_vectors );
	// make sure m_Across is correct size
	m_Across.resize( m_AcrossTangent.size() );

	update( i_Mesh );
}

//-------------------------------------------------------------------------------------------------
BBox GuideSet::getBounds() const
{
	// for now, recalculate on each request (could be cached with dirty flag etc)
	return grind::getBounds( m_P );
}

//-------------------------------------------------------------------------------------------------
void GuideSet::getData( const std::string& name, std::vector< Imath::V3f >& result ) const
{
	if( name == "P" ){
		m_P.getValue( result );
		return;
	}
	if( name == "Across" ){
		m_Across.getValue( result );
		return;
	}
	throw drd::RuntimeError( grindGetRiObjectName() + std::string( "trying to get vector data named '" ) + name + "'" );
}

//-------------------------------------------------------------------------------------------------
void GuideSet::setData( const std::string& name, const std::vector< Imath::V3f >& src )
{
	if( name == "P" ){
		m_P.setValue( src );
		return;
	}
	if( name == "Across" ){
		m_Across.setValue( src );
		return;
	}
	throw drd::RuntimeError( grindGetRiObjectName() + std::string( "trying to set vector data named '" ) + name + "'" );
}

//-------------------------------------------------------------------------------------------------
const DeviceVector< float >& GuideSet::getGuideLength() const
{
	// if the guide length hasn't been calculated, or the curve count has changed etc
	// this allows guide length to be requested even on subdivided guides etc
	if ( m_GuideLength.size() != m_CurveCount )
	{
		calcGuideLength();
	}
	assert( m_GuideLength.size() == m_CurveCount );
	return m_GuideLength;
}

//-------------------------------------------------------------------------------------------------
float GuideSet::getMaxGuideLength() const
{
	const DeviceVector< float >& guideLength = getGuideLength();

	float result = getMaxValue( guideLength );
	return result;
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
