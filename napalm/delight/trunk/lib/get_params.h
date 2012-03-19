#ifndef _NAPALM_DELIGHT_GET_PARAMS__H_
#define _NAPALM_DELIGHT_GET_PARAMS__H_

#include "napalmDelight/exceptions.h"

#include <vector>
#include <string>
#include <ri.h>

#include <napalm/core/Attribute.h>
#include <napalm/core/Table.h>
#include <napalm/core/TypedBuffer.h>


//! get a single named param from the object table
#define GET_REQUIRED_PARAM( o, param ) \
if ( !o.hasEntry( #param ) ) \
{ \
	throw NapalmDelightError( std::string("has no '") + std::string( #param ) + "' attribute"); \
	return; \
}\
if ( !o.getEntry( #param, param ) ) \
{ \
	throw NapalmDelightError( std::string("param '") + std::string( #param ) + "' has incorrect type"); \
	return; \
}


namespace napalm_delight
{
	//! param struct used for retaining data until Ri call has been made
	struct ParamStruct
	{
		// pointers to be passed through to rman
		std::vector< RtToken > tokens;
		std::vector< RtPointer > parms;

		//! data strides in token/parameter order
		// eg stride == 0: uniform attribute
		//    stride == 1: varying float
		//    stride == 3: varying V3f
		std::vector< int > strides;

		// token string storage
		std::vector< std::string > tbuffers;

		// value string array storage
		std::vector< std::vector< const char* > > stringVals;

		// contiguous arrays for varying data (fixed_ranges)
		std::vector< napalm::FloatBuffer::r_type > fbuffers;
		std::vector< napalm::V3fBuffer::r_type > vbuffers;

		// storage for constant data
		std::vector< float > fvals;
		std::vector< Imath::V3f > vvals;
	};


	//! fill a param struct based on an object table
	void getParams( const napalm::ObjectTable& o, ParamStruct& dst );
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
