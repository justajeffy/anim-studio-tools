
#include <drdDebug/log.h>
#include <iostream>
#include <boost/algorithm/string/split.hpp>

// vacuum includes
#include <vacuum/vacuum.hpp>
#include <vacuum/VacDataAccessor.hpp>
#include <vacuum/hdf5_conversions.hpp>

#include "VacLoader.h"
#include "gl/Texture.h"
#include "gl/Mesh.h"

//----------------------------------------------------------------------------
using namespace bee;
using namespace std;
DRD_MKLOGGER( L, "drd.bee.io.VacLoader" );

//----------------------------------------------------------------------------
namespace
{
	//----------------------------------------------------------------------------
	bool
	Separate
	(
		const string & a_In
	,	string & o_FilePart
	,	string & o_InternalPath
	,	char a_Delimiter = ':'
	)
	{
		/*
			m_FileName comes in as this:
			"/path/to/vac:internal/path:to/the/innner/stuff"
			And needs to get seperated into 2 components:
			m_FileName: "/path/to/vac"
			m_InternalPath: "internal/path:to/the/innner/stuff"
			Everything up to (but excluding) the first colon is the actual filename and
			everything from that first colon, excluding that first colon, but including every
			other colon until the end of the string.
		 */

		string::size_type delimiterPos = a_In.find( a_Delimiter );
		if( delimiterPos == string::npos )
		{
            // DRD_LOG_DEBUG( L, "Delimiter '" << a_Delimiter << "' not found in '" << a_In << "'" );
			return false;
		}

		o_FilePart = a_In.substr( 0, delimiterPos );
		o_InternalPath = a_In.substr( delimiterPos+1 );
        // DRD_LOG_DEBUG( L, "Separated out '" << a_In << "' into '" << o_FilePart << "' & '" << o_InternalPath << "' with delimiter '" << a_Delimiter << "'" );
		return true;
	}

	//----------------------------------------------------------------------------
    bool
    SeparateWithOverrides
    (
    	string const& a_FileParts
    ,	string& o_Filename
    ,	string& o_InternalPath
    ,	string& o_OverridesFilename
    ,	string& o_OverridesInternalPath
    )
    {
        string filePart, overridePart;
        bool hasOverrides = false;
        if( Separate( a_FileParts, filePart, overridePart, ';' ) )
        {
            if( !Separate( overridePart, o_OverridesFilename, o_OverridesInternalPath ) )
            {
            	DRD_LOG_ERROR( L, "Problem separating out overrides '" << overridePart << "'" );
                return false;
            }
            hasOverrides = true;
        }
        else
        {
            filePart = a_FileParts;
        }

        if( !Separate( filePart, o_Filename, o_InternalPath ) )
        {
            DRD_LOG_ERROR( L, "Problem separarting out '" << filePart << "'" );
            return false;
        }
        return hasOverrides;
    }

    //----------------------------------------------------------------------------
    bool
    SubIs( string const& a_FilePath )
    {
        string fileName;
        string internalPath;
        if( !Separate( string( a_FilePath ), fileName, internalPath ) )
        {
            return false;
        }

        transform( fileName.begin(), fileName.end(), fileName.begin(), ::tolower );
        return fileName.find( ".h5" ) == fileName.length() - 3;
    }

    //----------------------------------------------------------------------------
    template< class T >
    size_t
    getData
    (
    	char const* a_Name
    ,	float a_SampleTime
    ,	boost::shared_ptr< drd::VPrimPropertyList >& a_PropList
    ,	std::vector< T >& o_Data
    ,	bool const a_autoApplyTransforms
    ,	bool const a_AssertOnDimensionMismatch = true
    )
    {
    	size_t const vec_dim = sizeof( T ) / sizeof( float );

        int idx = a_PropList->getPropertyIndex( a_Name );
        if( idx == -1 )
        {
        	DRD_LOG_WARN( L, "Property '" << a_Name << "' was not found." );
            return 0;
        }

        if( a_SampleTime != 0.0f )
        {
            // if time is not exactly 0, then we're getting something animated, check to see if it IS animating.
            bool isAnimating = a_PropList->getPropertyIsAnimating( idx );
            if( !isAnimating )
            {
                return 0;
            }
        }

        // data does not belong to me
        float const* data_ptr = NULL;
        std::vector< float > temp_buf;
        drd::VPropertyInterpretation propInterp = drd::kInterpretNone;

        //----------------------------------------------------------------------------
        // read in the dataset dimensions so we know what we are going to be dealing with....
        std::vector< size_t > dimensions;
        getDimsFromAnimOrStaticParam( a_PropList->getPropertyParameter( idx ), dimensions );

        //----------------------------------------------------------------------------
        // if required to, warn about errors related to the dimensionality of the dataset
        if( a_AssertOnDimensionMismatch )
        {
        	if( dimensions[ 1 ] != vec_dim )
        	{
				DRD_LOG_WARN
				(
					L
				,	"Dimensions[ 1 ]'s stride is not what i expected, got: "
					<< dimensions[ 1 ]
					<< " : "
					<< vec_dim
				);
        	}
        }
        if( dimensions[ 1 ] != vec_dim )
        {
            return 0;
        }
        //----------------------------------------------------------------------------

        //----------------------------------------------------------------------------
        // read the actual data
        dimensions =
        	a_PropList->getProperty_float
        	(
        		idx
        	,	&data_ptr
        	,	temp_buf
        	,	a_SampleTime
        	,	&propInterp
        	,	a_autoApplyTransforms
        	);
        //----------------------------------------------------------------------------

        DRD_LOG_ASSERT( L, data_ptr, "data was NULL!: " << a_Name );

        size_t count = dimensions[ 0 ];
        T const* vertices = reinterpret_cast< T const* >( data_ptr );

        // if new memory was not allocated, then copy ( i.e. caching disabled ),
        // else rely on internal vac buffer storage, and trust that o_Data
        // has already been dealt with correctly.
    	if( count > 0 )
    	{
    		o_Data.resize( count );
    		std::copy( vertices, vertices + count, o_Data.begin() );
    	}

        return count;
    }

    //----------------------------------------------------------------------------
    size_t
    getData2or3
    (
    	char const* a_Name
    ,	float a_SampleTime
    ,	boost::shared_ptr< drd::VPrimPropertyList >& a_PropList
    ,	std::vector< Imath::V2f >& o_Data
    ,	bool const a_autoApplyTransforms
    ,	bool const a_AssertOnDimensionMismatch = true
    )
    {
        int count = getData( a_Name, a_SampleTime, a_PropList, o_Data, false );
        if( count == 0 )
        {
        	DRD_LOG_DEBUG( L, "Our data is NOT vec2, lets try with vec3!: " << a_Name );
            // this means that the was a mismatch in the dimensionality of the texCoords, which
            // we expected to be 2 (maya-esque), but was probably 3 (houdini-esque), so we
            // try again with vec3's instead of vec2's and then re-map them in (slow, but...)
            vector< Imath::V3f > texCoordVec3;
            count = getData( a_Name, a_SampleTime, a_PropList, texCoordVec3, a_autoApplyTransforms, true );

        	DRD_LOG_DEBUG( L, "Data still did not exist as vec3, must skip: " << a_Name );
        	if( count > 0 )
        	{
				o_Data.reserve( count );
				for( vector< Imath::V3f >::const_iterator it = texCoordVec3.begin() ; it != texCoordVec3.end() ; ++it )
				{
					Imath::V3f const& v = *it;
					o_Data.push_back( Imath::V2f( v.x, v.y ) );
				}
        	}
        }
        return count;
    }


  //----------------------------------------------------------------------------
    size_t
    getTopology
    (
    	drd::VacDataAccessor& a_ConnectivityAccessor
    ,	vector< MeshPolygon >& o_PolygonVector
    )
    {


        boost::shared_ptr< drd::VMeshTopology > topology =
        	boost::dynamic_pointer_cast< drd::VMeshTopology >
			(
				a_ConnectivityAccessor.getDataBlock()
			);

        int nfaces = topology->getNumFaceVertices().size();
        std::vector< int > const& nvertices = topology->getNumFaceVertices();
        std::vector< int > const& vertices = topology->getFaceVertices();

        DRD_LOG_ASSERT
        (
        	L
        ,	nfaces == nvertices.size()
        ,	"Obtained topology does not match: nfaces: "
			<< nfaces
			<< " != nverticies.size(): "
			<< nvertices.size()
        );

        DRD_LOG_INFO( L, "nfaces: " << nfaces );
        DRD_LOG_INFO( L, "nvertices.size(): " << nvertices.size() );
        DRD_LOG_INFO( L, "vertices.size(): " << vertices.size() );

        unsigned int faceIdx = 0;
        for( unsigned int faceCountIdx = 0 ; faceCountIdx < nvertices.size() ; ++faceCountIdx )
        {
            unsigned int faceCount = nvertices[ faceCountIdx ];

            DRD_LOG_ASSERT
            (
            	L
            ,	faceIdx + faceCount <= vertices.size()
            ,	"FaceCount out of bounds: ( faceIdx + faceCount ): "
				<< ( faceIdx + faceCount )
				<< " > vertices.size(): "
				<< vertices.size()
			);

            o_PolygonVector.push_back( MeshPolygon( faceCount ) );
            MeshPolygon& poly = o_PolygonVector.back();

            for( unsigned int cnt = 0 ; cnt < faceCount ; ++cnt )
            {
                unsigned idx_v = vertices[ faceIdx + cnt ];
                unsigned idx_n = vertices[ faceIdx + cnt ];
                unsigned idx_uv = faceIdx + cnt;
                poly.m_FaceVector.push_back( MeshFace( idx_v, idx_n, idx_uv ) );
            }

            faceIdx += faceCount;
        }
        return nvertices.size();
    }


}

//----------------------------------------------------------------------------
// static
bool
VacLoader::is( std::string const& a_Name )
{
    string filePart, overridePart;
	if( !Separate( a_Name, filePart, overridePart, ';' ) )
    {
        // no override part, that's cool, we'll just try without it.
        return SubIs( a_Name );
    }
    else
    {
        return SubIs( filePart ) && SubIs( overridePart );
    }
}

//----------------------------------------------------------------------------
VacLoader::VacLoader( std::string const& a_CombinedPath )
:	MeshLoader( a_CombinedPath, Loader::eVac )
,	m_Loaded( false )
,	m_PropListAccessor( NULL )
,   m_DeformListAccessor( NULL )
,	m_autoApplyTransforms( false )
,	m_referenceFrame( 0.0f )
,	m_referenceFrame_ptr( NULL )
{}

//----------------------------------------------------------------------------
VacLoader::~VacLoader()
{
    delete m_PropListAccessor;
    m_PropListAccessor = NULL;
    delete m_DeformListAccessor;
    m_DeformListAccessor = NULL;
}

//----------------------------------------------------------------------------
bool
VacLoader::open()
{
	if( m_Loaded )
	{
		DRD_LOG_WARN( L, "File already loaded: " + m_BaseFilename );
		return true;
	}

	DRD_LOG_ASSERT
	(
		L
	,	is( m_BaseFilename )
	,	"This is not a vacFile filePath, this was supposed to be checked before calling open(): "
		<< m_BaseFilename
	);

    m_HasOverrides = SeparateWithOverrides( m_BaseFilename, m_Filename, m_InternalPath, m_OverridesFilename, m_OverridesInternalPath );

	return load();
}

//----------------------------------------------------------------------------
bool
VacLoader::load()
{
	if( m_Loaded )
	{
		DRD_LOG_WARN( L, "File already loaded: " + m_BaseFilename );
		return true;
	}

	DRD_LOG_INFO( L, "Loading file '" + m_Filename + "' with internal path '" + m_InternalPath + "'" );
    if( m_HasOverrides )
    {
    	DRD_LOG_INFO( L, "\twith OVERRIDES file '" + m_OverridesFilename + "' with internal path '" + m_OverridesInternalPath + "'" );
    }
    else
    {
    	DRD_LOG_INFO( L, "\twith no overrides" );
    }

	// use vacuum error messages
	drd::UseVacuumHDF5Errors scopedObject;

	//==================================================
	// if file opened correctly, then read its contents
	//==================================================
	m_PropListAccessor = new drd::VacDataAccessor( 0, 1, true );
	std::string err;
	try
	{
		err = m_PropListAccessor->setPath( m_Filename, m_InternalPath, "__primProps" );
	}
	catch( ... )
	{
		DRD_LOG_ERROR( L, "Mesh internal path '" << m_InternalPath << "' is invalid for file '" << m_Filename << "'" );
		return false;
	}

	//================================
	// check if file correctly opened
	//================================
	if( !err.empty() )
	{
		DRD_LOG_INFO
		(
			L
		,	"Loading file '"
			<< m_Filename
			<< "' with internal path '"
			<< m_InternalPath
			<< "' had an error with the propList: "
			<< err
		);
		return false;
	}

	// TODO: getDataBlock throws
	boost::shared_ptr< drd::VPrimPropertyList > propList =
		boost::dynamic_pointer_cast< drd::VPrimPropertyList >
		(
			m_PropListAccessor->getDataBlock()
		);

	if( propList.get() == NULL )
	{
		DRD_LOG_ERROR( L, "propList.get() was NULL, and I was told that that's a very bad thing..." );
		return false;
	}

    // now for mesh topology
    drd::VacDataAccessor connectivityAccessor;
    try
    {
        err = connectivityAccessor.setPath( m_Filename, m_InternalPath, "__connectivity" );
    }
    catch( ... )
    {
    	DRD_LOG_ERROR
    	(
    		L
    	,	"Mesh internal path '"
			<< m_InternalPath
			<< "' is invalid for file '"
			<< m_Filename
			<< "', could not setPath to '__connectivity'!"
		);
        return false;
    }

    if( !err.empty() )
    {
    	DRD_LOG_ERROR
    	(
    		L
    	,	"Loading file '"
			<< m_Filename
			<< "' with internal path '"
			<< m_InternalPath
			<< "' had an error with the '__connectivity': "
			<< err
		);
        return false;
    }

	// TODO: what time is love?
	float sampleTime = 0.0f;

    {
        size_t count = 0;

		// we don't want to apply any transform to the rest pose (m_StaticVertextVector etc)
    	m_referenceFrame_ptr = NULL;

    	count = getData( "P", sampleTime, propList, m_StaticVertexVector, false );
    	if ( ( count == 0 ) && ( m_StaticVertexVector.empty() ) ) DRD_LOG_ERROR( L, "Error no P in getData(P) for " + m_BaseFilename );
    	count = getData( "N", sampleTime, propList, m_StaticNormalVector, false );
    	if ( ( count == 0 ) && ( m_StaticNormalVector.empty() ) ) DRD_LOG_ERROR( L, "Error no N in in getData(N) for " + m_BaseFilename );
        count = getData2or3( "uv", sampleTime, propList, m_StaticTexCoordVector, false, false );
        if ( ( count == 0 ) && ( m_StaticTexCoordVector.empty() ) ) DRD_LOG_ERROR( L, "Error no UV in getData(uv) for " + m_BaseFilename );

        count = getTopology( connectivityAccessor, m_StaticPolygonVector );
    }

    if( m_HasOverrides )
    {
        m_DeformListAccessor = new drd::VacDataAccessor( 0, 1, true );

        //========================================================
        // deform the position data by the input deformation data
        //========================================================
        std::string deform_err;

        try
        {
            deform_err = m_DeformListAccessor->setPath( m_OverridesFilename, m_OverridesInternalPath, "__primProps" );
        }
        catch( ... )
        {
        	DRD_LOG_ERROR( L, "deform internal path '" << m_OverridesInternalPath << "' is invalid" );
            return false;
        }

        if( !deform_err.empty() )
        {
        	DRD_LOG_ERROR( L, "Some error: " << deform_err );
            return false;
        }

        // TODO: getDataBlock throws
        boost::shared_ptr< drd::VPrimPropertyList > deformPropList =
            boost::dynamic_pointer_cast< drd::VPrimPropertyList >
			(
				m_DeformListAccessor->getDataBlock()
			);

        //=======================================================
        // Apply the deforming data on top of the static data
        //=======================================================
        propList->override( *deformPropList );
    }

	// "vertex" data read in
    {
        size_t count = 0;
    	if( m_autoApplyTransforms && m_referenceFrame_ptr )
    	{
    		DRD_LOG_DEBUG( L, "VacLoader::load setReferenceFrame: " << m_referenceFrame );
    		propList->setReferenceFrame( m_referenceFrame );
    	}
        count = getData( "P", sampleTime, propList, m_VertexVector, m_autoApplyTransforms );
        if ( ( count == 0 ) && ( m_StaticVertexVector.empty() ) ) DRD_LOG_WARN( L, "Error no P in getData(P) for " + m_BaseFilename );
        count = getData( "N", sampleTime, propList, m_NormalVector, m_autoApplyTransforms );
        if ( ( count == 0 ) && ( m_StaticNormalVector.empty() ) ) DRD_LOG_WARN( L, "Error no N in getData(N) for " + m_BaseFilename );
        count = getData2or3( "uv", sampleTime, propList, m_TexCoordVector, m_autoApplyTransforms, false );
        if ( ( count == 0 ) && ( m_StaticTexCoordVector.empty() ) ) DRD_LOG_WARN( L, "Error no UV in getData(uv) for " + m_BaseFilename );
        count = getTopology( connectivityAccessor, m_PolygonVector );
    }

    DRD_LOG_DEBUG( L, "bee::VacLoader::m_autoApplyTransforms = " << m_autoApplyTransforms );
    DRD_LOG_DEBUG( L, "bee::VacLoader::m_referenceFrame = " << m_referenceFrame );

	m_Loaded = true;

	return true;
}

//----------------------------------------------------------------------------
bool
VacLoader::write()
{
	throw std::runtime_error( "Writing is not yet supported" );
	return false;
}

//----------------------------------------------------------------------------
bool
VacLoader::close()
{
	delete m_PropListAccessor;
	m_PropListAccessor = NULL;
}

//----------------------------------------------------------------------------
void
VacLoader::reportStats()
{
	DRD_LOG_INFO( L, "VacLoader Report :" );
	DRD_LOG_INFO
	(
		L
	,	"- Filename '"
		<< m_Filename
		<< "' with internal path '"
		<< m_InternalPath
		<< "'"
	);
    if( m_HasOverrides )
    {
    	DRD_LOG_INFO
    	(
    		L
    	,	"-     Overrides Filename '"
			<< m_OverridesFilename
			<< "' with internal path '"
			<< m_OverridesInternalPath
			<< "'"
		);
    }
    DRD_LOG_INFO( L, "- vertex count : " << m_VertexVector.size() );
    DRD_LOG_INFO( L, "- normal count : " << m_NormalVector.size() );
    DRD_LOG_INFO( L, "- texCoord count : " << m_TexCoordVector.size() );
    DRD_LOG_INFO( L, "- polygon count : " << m_PolygonVector.size() );
}

//----------------------------------------------------------------------------
boost::shared_ptr< Mesh >
VacLoader::createMesh()
{
	DRD_LOG_ASSERT( L, false, "Method not implemented." );
	return boost::shared_ptr< Mesh >();
}

//----------------------------------------------------------------------------
//! set automatic application of transforms for vac files
void
VacLoader::setAutoApplyTransforms( bool const a_autoApplyTransforms )
{
	DRD_LOG_DEBUG( L, "bee VacLoader::setAutoApplyTransforms( " << a_autoApplyTransforms << " )" );
	m_autoApplyTransforms = a_autoApplyTransforms;
}

//----------------------------------------------------------------------------
//! get status of automatic application of transforms for vac files
bool const
VacLoader::getAutoApplyTransforms() const
{
	return m_autoApplyTransforms;
}

//----------------------------------------------------------------------------
//! get the reference frame to use for a vac file ( only of use for rendering )
//! - specifies a frame that all other frames are transformed into, used for
//! - the determination of local space
void VacLoader::setReferenceFrame( float const a_frame )
{
	DRD_LOG_DEBUG( L, "bee VacLoader::setReferenceFrame( " << a_frame << " )" );
	m_referenceFrame = a_frame;
	m_referenceFrame_ptr = &m_referenceFrame;
}

//----------------------------------------------------------------------------
bool
VacLoader::isAnimatable()
{
	return true;
}

//----------------------------------------------------------------------------
bool
VacLoader::setFrame( float a_Frame )
{
	// TODO: getDataBlock throws
	boost::shared_ptr< drd::VPrimPropertyList > propList =
		boost::dynamic_pointer_cast< drd::VPrimPropertyList >
		(
			m_PropListAccessor->getDataBlock()
		);

	if( propList.get() == NULL )
	{
		// TODO: Bad things
		DRD_LOG_WARN( L, "Was NOT able to set frame to: " << a_Frame );
		return false;
	}

	// set the reference frame if there is one....
	if( m_autoApplyTransforms && m_referenceFrame_ptr )
	{
		DRD_LOG_DEBUG( L, "VacLoader::setFrame setReferenceFrame: " << m_referenceFrame );
		propList->setReferenceFrame( m_referenceFrame );
	}

	size_t count = 0;

	getData( "P", a_Frame, propList, m_VertexVector, m_autoApplyTransforms );
	if ( ( count == 0 ) && ( m_StaticVertexVector.empty() ) ) DRD_LOG_ERROR( L, "Error no P in getData(P) for " + m_BaseFilename );
	getData( "N", a_Frame, propList, m_NormalVector, m_autoApplyTransforms );
	if ( ( count == 0 ) && ( m_StaticNormalVector.empty() ) ) DRD_LOG_ERROR( L, "Error no N in getData(N) for " + m_BaseFilename );
    getData2or3( "uv", a_Frame, propList, m_TexCoordVector, m_autoApplyTransforms, false );
    if ( ( count == 0 ) && ( m_StaticTexCoordVector.empty() ) ) DRD_LOG_ERROR( L, "Error no UV in getData(uv) for " + m_BaseFilename );

    DRD_LOG_DEBUG( L, "Set frame to: " << a_Frame );

	return true;
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
