#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:io" );

#include "io_ptc.h"

#include "napalmDelight/io.h"
#include "napalmDelight/exceptions.h"

using namespace napalm_delight;
namespace n = napalm;

namespace napalm_delight {

//-------------------------------------------------------------------------------------------------
n::object_table_ptr load( const std::string& i_path )
{
	std::string ext = i_path.substr( i_path.find_last_of(".") + 1 );
	DRD_LOG_DEBUG( L, "io dispatch extension: " << ext );

	// dispatch the appropriate IO
	if( ext == "ptc" )
		return loadPtc( i_path );
	else
		throw NapalmDelightError( std::string( "file extension '" + ext + "' not yet supported" ) );
}


//-------------------------------------------------------------------------------------------------
void save( const std::string& i_path, const napalm::ObjectTable& t )
{
	std::string ext = i_path.substr( i_path.find_last_of(".") + 1 );
	DRD_LOG_DEBUG( L, "io dispatch extension: " << ext );

	// dispatch the appropriate IO
	if( ext == "ptc" )
		return savePtc( i_path, t );
	else
		throw NapalmDelightError( std::string( "file extension '" + ext + "' not yet supported" ) );
}

}
