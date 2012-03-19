#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:curves" );

#include "get_params.h"

#include <napalmDelight/render.h>

#include <napalm/core/Attribute.h>

using namespace napalm;
using namespace napalm_delight;


void napalm_delight::curves( const ObjectTable& t )
{
	std::string type, wrap;
	GET_REQUIRED_PARAM( t, type );
	GET_REQUIRED_PARAM( t, wrap );

	IntBufferCPtr nvertices;
	GET_REQUIRED_PARAM( t, nvertices );
	int ncurves = nvertices->size();

	if( t.hasEntry("firstN" ) ){
		if( !t.getEntry( "firstN", ncurves ) ){
			throw NapalmDelightError( std::string( "firstN variable should be int" ) );
		}
	}

	ParamStruct p;
	getParams( t, p );

	RiCurvesV( type.c_str(), ncurves, const_cast<RtInt*>(&(nvertices->r()[0])), wrap.c_str(), p.tokens.size(), &(p.tokens[0]), &(p.parms[0]) );
}

