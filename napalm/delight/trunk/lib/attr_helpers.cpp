


#include "napalmDelight/attr_helpers.h"

#include "napalmDelight/exceptions.h"

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

using namespace napalm;
using namespace Imath;

namespace napalm_delight
{

//-------------------------------------------------------------------------------------------------
template< typename T >
T getAttr( napalm::c_object_table_ptr t, const char* name )
{
	if( t->hasEntry(name) ){
		T val0;
		if( t->getEntry( name, val0 ) )
			return val0;
		else
			throw NapalmDelightError( std::string( "user attribute " ) + name + " has incorrect type" );
	} else {
		throw NapalmDelightError( std::string("no attribute ") + name  + " defined.  Has a default been set?" );
	}
}

//-------------------------------------------------------------------------------------------------
template< >
float getAttr( 	napalm::c_object_table_ptr t,
				const char* name )
{
	if ( t->hasEntry( name ) )
	{
		float val0;
		int val1;
		if ( t->getEntry( name, val0 ) ) return val0;
		else if ( t->getEntry( name, val1 ) ) return val1;
		else throw NapalmDelightError( std::string( "user attribute " ) + name + " has incorrect type" );
	}
	else
	{
		throw NapalmDelightError( std::string( "no attribute " ) + name + " defined.  Has a default been set?" );
	}
}
}

//template float napalm_delight::getAttr( napalm::c_object_table_ptr t, const char* name );
template int napalm_delight::getAttr( napalm::c_object_table_ptr t, const char* name );
template std::string napalm_delight::getAttr( napalm::c_object_table_ptr t, const char* name );
template V3f napalm_delight::getAttr( napalm::c_object_table_ptr t, const char* name );
template M44f napalm_delight::getAttr( napalm::c_object_table_ptr t, const char* name );
