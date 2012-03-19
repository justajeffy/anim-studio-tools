#include <drdDebug/log.h>
DRD_MKLOGGER( L, "napalmDelight:render" );

#include <napalmDelight/render.h>

#include <ri.h>
#include <rx.h>


//-------------------------------------------------------------------------------------------------
bool napalm_delight::hasRenderManContext()
{
	// according to renderman spec...
	// RiGetContext returns a handle for the current active rendering context.
	// If there is no active rendering context, RI_NULL will be returned
	return RiGetContext() != RI_NULL;
}

//-------------------------------------------------------------------------------------------------
namespace napalm_delight
{
	bool getRiAttribute( std::string const& attribute_name, std::string& result )
	{
		// get the current shutter time info from the rib file
		RxInfoType_t o_result_type;
		RtInt o_result_count;
		RtString o_data[ 1 ];

		// let the user know if the shutter was found
		bool success = ( 0 == RxAttribute
								(
									attribute_name.c_str()
								,	o_data
								,	sizeof( RtString[ 1 ] )
								,	&o_result_type
								,	&o_result_count
								)
						);

		if( o_data && success )
		{
			result = o_data[ 0 ];
		}
		else
		{
			result.clear();
		}

		return success;
	}
}

//-------------------------------------------------------------------------------------------------
std::string napalm_delight::renderManObjectName()
{
	std::string result;
	if ( napalm_delight::hasRenderManContext() )
	{
		napalm_delight::getRiAttribute( "identifier:name", result );
	}
	return result;
}
