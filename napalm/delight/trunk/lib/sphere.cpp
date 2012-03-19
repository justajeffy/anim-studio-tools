#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:sphere" );

#include "get_params.h"

#include <napalmDelight/render.h>

#include <napalm/core/Attribute.h>

using namespace napalm;
using namespace napalm_delight;

void napalm_delight::sphere( const ObjectTable& o )
{
	float radius, zmin, zmax, thetamax;

	GET_REQUIRED_PARAM( o, radius );
	GET_REQUIRED_PARAM( o, zmin );
	GET_REQUIRED_PARAM( o, zmax );
	GET_REQUIRED_PARAM( o, thetamax );

	ParamStruct p;
	getParams( o, p );

	RiSphereV( radius, zmin, zmax, thetamax, p.tokens.size(), &(p.tokens[0]), &(p.parms[0]) );
}

