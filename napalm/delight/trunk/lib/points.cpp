#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:points" );

#include "get_params.h"

#include "napalmDelight/render.h"
#include "napalmDelight/exceptions.h"

#include <napalm/core/Attribute.h>

using namespace napalm;
using namespace napalm_delight;


void napalm_delight::points( const ObjectTable& t )
{
	V3fBufferCPtr P;
	GET_REQUIRED_PARAM( t, P );

	int n = P->size();

	if( t.hasEntry("firstN" ) ){
		if( !t.getEntry( "firstN", n ) ){
			throw NapalmDelightError( std::string( "firstN variable should be int" ) );
		}
	}

	assert( n <= P->size() );

	ParamStruct p;
	getParams( t, p );

	DRD_LOG_DEBUG( L, "emitting " << n << " RiPoints" );
	RiPointsV( n, p.tokens.size(), &(p.tokens[0]), &(p.parms[0]) );
}

