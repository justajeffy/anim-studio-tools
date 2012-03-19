#ifndef _NAPALM_DELIGHT_ATTR_HELPERS__H_
#define _NAPALM_DELIGHT_ATTR_HELPERS__H_

#include <napalm/core/Table.h>

namespace napalm_delight
{

//! set the default value for an attribute
template< typename T >
void setAttrDefault( napalm::object_table_ptr t, const char* name, T val )
{
	if( !t->hasEntry(name) )
		t->setEntry( name, val );
}

//! get an attribute of a specific type from the params table
template< typename T >
T getAttr( napalm::c_object_table_ptr t, const char* name );


}

#endif


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
