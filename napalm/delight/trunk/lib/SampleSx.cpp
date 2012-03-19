#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:SampleSx" );

#include "napalmDelight/SampleSx.h"
#include "napalmDelight/attr_helpers.h"
#include "napalmDelight/type_conversion.h"
#include "napalmDelight/render.h"
#include "napalmDelight/exceptions.h"

#include <napalm/core/Table.h>
#include <napalm/core/TypedBuffer.h>

#include <boost/variant/get.hpp>

namespace n = napalm;
using namespace napalm_delight;
using namespace drd;

//-------------------------------------------------------------------------------------------------------------
template< typename T >
bool tryToGrab( n::object_table_ptr entries,
				std::vector< T >& result )
{
	n::ObjectTable::const_iterator i( entries->begin() ), e( entries->end() );

	result.clear();
	result.reserve( entries->size() );

	for ( ; i != e ; ++i )
	{
		T val;
		if ( !entries->getEntry( ( *i ).first, val ) ) return false;
		result.push_back( val );
	}
//	DRD_LOG_DEBUG( L, "found " << result.size() << " entries" );
	return true;
}

//-------------------------------------------------------------------------------------------------------------
SampleSx::SampleSx( napalm::c_object_table_ptr params, ::SxContext i_parent ) :
	m_userParams( params ), m_sx( new SxContextWrapper( i_parent ) )
{
#if 0
	// NOTE: this should no longer be required due
	//       as SxContextWrapper now supports querying current context
	// need to set render:nthreads if thread count > core count
	// refer 3delight release notes
	float renderStateThreads = 8;
	if( getRxOption( "render:nthreads", renderStateThreads ) )
	{
		std::vector< int > nthreads;
		nthreads.push_back( int(renderStateThreads) );
		DRD_LOG_DEBUG( L, "setting sx context threads to " << renderStateThreads );
		SxSetOption( m_sx, "render:nthreads", nthreads );
	}
#endif

	bool hasRMan = hasRenderManContext();

	if ( !hasRMan )
	{
		DRD_LOG_DEBUG( L, "no renderman context detected, setting up basic matrices" );
		// todo: query the current shader search path using RxOption
		std::vector< char* > searchPath;
		searchPath.push_back( "@:." );
		SxSetOption( m_sx, "searchpath:shader", searchPath );

		float mtx[] =
		{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
		std::vector< float > identityMatrix;
		identityMatrix.assign( mtx, mtx + 16 );

		SxDefineSpace( m_sx, "shader", identityMatrix );
		SxDefineSpace( m_sx, "object", identityMatrix );
		SxDefineSpace( m_sx, "world", identityMatrix );
	}
	#if 0
	 else {
		DRD_LOG_VERBATIM( L, "setting up matrices" );

		float mtx[] =
		{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1 };
		std::vector< float > myMtx(16);
#if 0
		myMtx.assign( mtx, mtx + 16 );

		//SxDefineSpace( m_sx, "object", myMtx );

		myMtx[14] = -1;
#endif

		RtMatrix temp;
		RxTransform( "object", "world", 0.0f, temp );
		int ii = 0;
		for( int i = 0; i < 4; ++i ){
			for( int j = 0; j < 4; ++j, ++ii ){
				myMtx[ ii ] = temp[i][j];
				std::cerr << myMtx[ii] << " ";
			}
			std::cerr << "\n";
		}



//		SxDefineSpace( m_sx, "current", myMtx );
		SxDefineSpace( m_sx, "object", myMtx );

	}
	#endif

	drd::SxParameterList parameterList = SxCreateParameterList( m_sx, 1, "object" );
	m_shaderName = getAttr< std::string > ( m_userParams, "name" );

	if( m_shaderName.empty() )
		throw NapalmDelightError( "empty string provided for shader name" );

	DRD_LOG_DEBUG( L, "creating shader with name '" << m_shaderName << "'" );
	m_shader = SxCreateShader( m_sx, parameterList, m_shaderName, m_shaderName );

	if( !m_shader || !m_shader.get()->is_good() )
		throw NapalmDelightError( std::string( "failed to create shader (possible bad file path): " ) + m_shaderName );

	n::object_table_ptr shaderParams;
	// if there are any shader params
	if ( m_userParams->getEntry( "params", shaderParams ) )
	{
		n::ObjectTable::const_iterator i( shaderParams->begin() ), e( shaderParams->end() );
		// for each shader param
		for ( ; i != e ; ++i )
		{
			assert( i.getKey<std::string>() != NULL );
			const std::string& paramName = *i.getKey<std::string>();

			n::object_table_ptr entries;
			if ( !i.getValue( entries ) ) throw NapalmDelightError( std::string( "can't get values for param " ) + paramName );

			std::vector< float > floatParams;
			std::vector< int > intParams;
			std::vector< std::string > stringParams;

			if ( tryToGrab( entries, floatParams ) )
			{
				DRD_LOG_DEBUG( L, "setting float shader param '" << paramName << "'" );
				//SxSetParameter( parameterList, paramName, SxFloat, floatParams, false );
				drd::SxSetParameter( m_shader, parameterList, paramName, floatParams, false );
			}
			if ( tryToGrab( entries, intParams ) )
			{
				DRD_LOG_DEBUG( L, "setting int shader param '" << paramName << "'" );
				//SxSetParameter( parameterList, paramName, SxInt, intParams, false );
				drd::SxSetParameter( m_shader, parameterList, paramName, intParams, false );
			}
			if ( tryToGrab( entries, stringParams ) )
			{
				DRD_LOG_DEBUG( L, "setting string shader param '" << paramName << "'" );
				std::vector< char* > sparms( stringParams.size() );
				for ( size_t i = 0 ; i < stringParams.size() ; ++i )
					sparms[ i ] = const_cast< char* > ( stringParams[ i ].c_str() );
				//SxSetParameter( parameterList, paramName, SxString, sparms, false );
				drd::SxSetParameter( m_shader, parameterList, paramName, sparms, false );
			}
		}
	}

	// now that we've set up the parameter list, the real shader is created
	m_shader = SxCreateShader( m_sx, parameterList, m_shaderName, m_shaderName );
}

//-------------------------------------------------------------------------------------------------------------
void SampleSx::shaderInfo() const
{
	unsigned int numParams = SxGetNumParameters( m_shader );
	DRD_LOG_VERBATIM( L, "'" << m_shaderName << "' params: " << numParams << "\n" << m_userParams );
}

//-------------------------------------------------------------------------------------------------------------
bool SampleSx::sample(	const n::ObjectTable& i_sampleParams,
                      	const n::ObjectTable& i_aovs,
						n::ObjectTable& o_data ) const
{
	assert( m_shader != NULL );
	assert( m_sx != NULL );
	// try to derive array size from "P"
	n::V3fBufferPtr P_temp;
	if ( !i_sampleParams.getEntry( "P", P_temp ) ) throw NapalmDelightError( "unable to access 'P'" );
	const size_t sampleCount = P_temp->size();

	drd::SxParameterList parameterList = SxCreateParameterList( m_sx, sampleCount, "object" );

	for ( n::ObjectTable::const_iterator i( i_sampleParams.begin() ), e( i_sampleParams.end() ) ; i != e ; ++i )
	{
		assert( i.getKey<std::string>() != NULL );
		const std::string& paramName = *i.getKey<std::string>();

		// floats
		{
			n::FloatBufferPtr buf;
			if ( i.getValue( buf ) )
			{
				assert( buf->size() == sampleCount );
				n::FloatBuffer::r_type fr = buf->r();
				// todo: remove intermediate copy
				std::vector< float > temp( buf->size() );
				std::copy( fr.begin(), fr.end(), temp.begin() );
				DRD_LOG_DEBUG( L, "setting float param '" << paramName << "'" );
				drd::SxSetParameter( m_shader, parameterList, paramName, temp, true );
			}
		}

		// V3f
		{
			n::V3fBufferPtr buf;
			if ( i_sampleParams.getEntry( ( *i ).first, buf ) )
			{
				assert( buf->size() == sampleCount );
				n::V3fBuffer::r_type fr = buf->r();
				// todo: remove intermediate copy
				std::vector< float > temp( buf->size() * 3 );
				std::copy( reinterpret_cast< const float* > ( &*fr.begin() ), reinterpret_cast< const float* > ( &*fr.end() ), temp.begin() );
				DRD_LOG_DEBUG( L, "setting V3f param '" << paramName << "'" );
				drd::SxSetParameter( m_shader, parameterList, paramName, temp, true );
			}
		}
	}

	DRD_LOG_DEBUG( L, "sampling shader '" << m_shaderName << "' (" << sampleCount << " samples)" );
	if ( !SxCallShader( m_shader, parameterList ) ) throw NapalmDelightError( std::string( "error evaluating shader: " ) + m_shaderName );

	bool success = false;

	// for each requested aov
	for ( n::ObjectTable::const_iterator i( i_aovs.begin() ), e( i_aovs.end() ) ; i != e ; ++i )
	{
		std::string aovName;
		i_aovs.getEntry( ( *i ).first, aovName );
		DRD_LOG_DEBUG( L, "trying to sample aov: " << aovName );

		drd::SxData data;
		if ( SxGetParameter( m_shader, parameterList, aovName, data ) )
		{
			const std::vector< float >& floatData = boost::get< std::vector< float > >( data );

			if ( floatData.size() == sampleCount )
			{
				n::FloatBufferPtr buf( new n::FloatBuffer() );
				buf->resize( sampleCount );
				n::FloatBuffer::w_type fr = buf->w();
				std::copy( floatData.begin(), floatData.end(), fr.begin() );
				DRD_LOG_DEBUG( L, "retrieved " << buf->size() << " float values for " << aovName );
				o_data.setEntry( aovName, buf );
				success = true;
			}

			else if ( floatData.size() == sampleCount * 3 )
			{
				n::V3fBufferPtr buf( new n::V3fBuffer() );
				buf->resize( sampleCount );
				n::V3fBuffer::w_type fr = buf->w();
				std::copy( (Imath::V3f*) &*floatData.begin(), (Imath::V3f*) &*floatData.end(), fr.begin() );
				DRD_LOG_DEBUG( L, "retrieved " << buf->size() << " V3f values for " << aovName );
				o_data.setEntry( aovName, buf );
				success = true;
			}

			else
			{
				throw NapalmDelightError( "buffer has incorrect size" );
			}
		} else {
			throw NapalmDelightError( std::string("failed to get aov: '") + aovName + "'" );
		}
	}

	return success;
}

//-------------------------------------------------------------------------------------------------------------
bool SampleSx::sample(	const n::ObjectTable& i_sampleParams,
						n::ObjectTable& o_data ) const
{
	n::c_object_table_ptr aovs;
	if( !m_userParams->getEntry( "aovs", aovs ) ){
		throw NapalmDelightError( "tried to sample without naming aovs" );
	}
	return sample( i_sampleParams, *aovs, o_data );
}

//-------------------------------------------------------------------------------------------------------------
std::string SampleSx::name() const
{
	std::string name;
	if( !m_userParams->getEntry( "name", name ) )
		throw NapalmDelightError( "error querying shader name" );
	return name;
}

//-------------------------------------------------------------------------------------------------------------
SampleSx::~SampleSx()
{}

