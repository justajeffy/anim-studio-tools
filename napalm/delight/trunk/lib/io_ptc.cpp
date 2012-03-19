#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:io" );

#include "io_ptc.h"

#include "napalmDelight/io.h"
#include "napalmDelight/exceptions.h"

#include <napalm/core/TypedBuffer.h>

#include "pointcloud.h"

using namespace napalm_delight;
namespace n = napalm;


//-------------------------------------------------------------------------------------------------
// to automate file closing
struct ScopePtcReadFile
{
	ScopePtcReadFile( const char* i_path )
	{
		data = PtcSafeOpenPointCloudFile( i_path );
	}

	~ScopePtcReadFile()
	{
		if( data != NULL )
			PtcClosePointCloudFile( data );
	}

	PtcPointCloud data;
};

//-------------------------------------------------------------------------------------------------
inline int getDataSize( const std::string& input )
{
	if( input == "float" )
		return 1;
	else if( input == "integer" )
		return 1;
	else if( input == "point" )
		return 3;
	else if( input == "color" )
		return 3;
	else if( input == "normal" )
		return 3;
	else if( input == "vector" )
		return 3;
	else if( input == "matrix" )
		return 16;
	else
		return 0;
}

//-------------------------------------------------------------------------------------------------
typedef n::FloatBuffer::w_type::iterator f_iter;
typedef n::V3fBuffer::w_type::iterator V3f_iter;
typedef boost::variant< f_iter, V3f_iter > iter_variants;

//-------------------------------------------------------------------------------------------------
struct StoreValueVisitor: public boost::static_visitor< void >
{
	StoreValueVisitor( float* ptr ) :
		m_ptr( ptr )
	{
	}

	void operator()( f_iter& iter ) const
	{
		*iter++ = *m_ptr;
	}

	void operator()( V3f_iter& iter ) const
	{
		*iter++ = *( reinterpret_cast< Imath::V3f* > ( m_ptr ) );
	}

private:
	float* m_ptr;
};

//-------------------------------------------------------------------------------------------------
template< typename BUFPTR >
void setUpBuffer( const int npoints, const char* name, const char* type, n::object_table_ptr& result, std::vector< iter_variants >& iters )
{
	typedef typename BUFPTR::value_type BUF;
	BUFPTR buf( new BUF( npoints ) );

	// a token so this data can be passed directly to eg renderman points
	std::string token = std::string( "varying " ) + type + " " + name;
	DRD_LOG_DEBUG( L, "token: " << token );
	n::attrib_table_ptr attribs = buf->getAttribs();
	attribs->setEntry( "token", token );
	// store the raw type
	attribs->setEntry( "type", type );
	result->setEntry( name, buf );
	iters.push_back( buf->w().begin() );
}

//-------------------------------------------------------------------------------------------------
n::object_table_ptr napalm_delight::loadPtc( const std::string& i_path )
{
	ScopePtcReadFile pc( i_path.c_str() );

	if( pc.data == NULL )
		throw NapalmDelightError( std::string("invalid point cloud file: ") + i_path );

	int npoints;
	int nvars;
	int userdatasize;
	const char** names = NULL;
	const char** types = NULL;
	//std::vector<float> world2eye(16), world2ndc(16);
	Imath::M44f world2eye, world2ndc;

	PtcGetPointCloudInfo( pc.data, "npoints", &npoints );
	PtcGetPointCloudInfo( pc.data, "nvars", &nvars );
	PtcGetPointCloudInfo( pc.data, "varnames", &names );
	PtcGetPointCloudInfo( pc.data, "vartypes", &types );
	PtcGetPointCloudInfo( pc.data, "datasize", &userdatasize );
	PtcGetPointCloudInfo( pc.data, "world2eye", &world2eye[0][0] );
	PtcGetPointCloudInfo( pc.data, "world2ndc", &world2ndc[0][0] );

	DRD_LOG_DEBUG( L, "npoints: " << npoints );
	DRD_LOG_DEBUG( L, "datasize: " << userdatasize );
	DRD_LOG_DEBUG( L, "nvars: " << nvars );
	DRD_LOG_DEBUG( L, "world2eye:\n" << world2eye );
	DRD_LOG_DEBUG( L, "world2ndc:\n" << world2ndc );

	n::object_table_ptr result( new n::ObjectTable() );

	result->setEntry( "world2eye", world2eye );
	result->setEntry( "world2ndc", world2ndc );

	// P
	n::V3fBufferPtr P_buf( new n::V3fBuffer( npoints ) );
	P_buf->getAttribs()->setEntry( "token", "P" );
	result->setEntry( "P", P_buf );
	V3f_iter P_iter = P_buf->w().begin();

	// N
	n::V3fBufferPtr N_buf( new  n::V3fBuffer( npoints ) );
	N_buf->getAttribs()->setEntry( "token", "N" );
	result->setEntry( "N", N_buf );
	V3f_iter N_iter = N_buf->w().begin();

	// radius
	n::FloatBufferPtr r_buf( new  n::FloatBuffer( npoints ) );
	r_buf->getAttribs()->setEntry( "token", "varying float width" );
	result->setEntry( "radius", r_buf );
	f_iter r_iter = r_buf->w().begin();

	// set up buffers, iterators and offsets
	std::vector<int> offsets;
	std::vector< iter_variants > iters;
	int offset = 0;
	for( int i = 0; i < nvars; ++i )
	{
		DRD_LOG_DEBUG( L, "var[" << i << "]: " << names[i] << " type: " << types[i] );
		int elemSize = getDataSize( types[i] );
		if ( elemSize == 1 )
		{
			setUpBuffer< n::FloatBufferPtr > ( npoints, names[ i ], types[ i ], result, iters );
			offsets.push_back( offset );
		}
		else if ( elemSize == 3 )
		{
			setUpBuffer< n::V3fBufferPtr > ( npoints, names[ i ], types[ i ], result, iters );
			offsets.push_back( offset );
		}
		else
		{
			DRD_LOG_WARN( L, "skipping due to unsupported data size: " << names[i] );
		}

		offset += elemSize;
	}

	assert( nvars == offsets.size() );
	assert( nvars == iters.size() );

	// temp buffers for outputs from delight
	Imath::V3f point, normal;
	float radius;
	std::vector< float > user_data( userdatasize );

	// for each point
	for( int i = 0; i < npoints; ++i )
	{
		if( 0 == PtcReadDataPoint( pc.data, reinterpret_cast<float*>( &point.x ), reinterpret_cast<float*>( &normal.x), &radius, &user_data[0] ) )
			throw NapalmDelightError( std::string( "error reading point data from: " ) + i_path );

		// store the standard data
		*(P_iter++) = point;
		*(N_iter++) = normal;
		*(r_iter++) = radius;

		// now the user data
		std::vector< iter_variants >::iterator buf_iter( iters.begin() );
		std::vector< int >::const_iterator os_iter( offsets.begin() );
		for( int j = 0; j < nvars; ++j, ++os_iter, ++buf_iter )
		{
			boost::apply_visitor( StoreValueVisitor( &user_data[ *os_iter ] ), *buf_iter );
		}
	}

	return result;
}


//-------------------------------------------------------------------------------------------------
void napalm_delight::savePtc( const std::string& i_path, const napalm::ObjectTable& t )
{
	throw NapalmDelightError( "savePtc not implemented yet" );
}

