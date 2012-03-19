/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: utils.cpp 88551 2011-06-24 07:55:38Z luke.emrose $"
 */

#include "utils.h"
#include "context.h"
#include <rx.h>

//-------------------------------------------------------------------------------------------------
namespace grind
{
	using namespace std;

	//-------------------------------------------------------------------------------------------------
	// couldn't get bindings to work directly on these
	bool hasGPU(){ return ContextInfo::instance().hasGPU(); }
	bool hasEmulation(){ return ContextInfo::instance().hasEmulation(); }
	bool hasOpenGL(){ return ContextInfo::instance().hasOpenGL(); }
	bool hasRX(){ return ContextInfo::instance().hasRX(); }

	//----------------------------------------------------------------------------
	/**
	 * get a string attribute from the current scene stack
	 */
	bool grindGetRiAttribute( string const& attribute_name, string& result )
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

	//----------------------------------------------------------------------------
	//! if we are in renderman then return the value of the identified name attached to this object
	string grindGetRiObjectName()
	{
		string result;
		if( hasRX() ) { grindGetRiAttribute( "identifier:name", result ); }
		return result + ": ";
	}
}


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
