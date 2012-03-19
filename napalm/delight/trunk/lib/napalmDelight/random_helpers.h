#ifndef _NAPALM_DELIGHT_RANDOM_HELPERS__H_
#define _NAPALM_DELIGHT_RANDOM_HELPERS__H_

#include <OpenEXR/ImathVec.h>

namespace napalm_delight
{

//-------------------------------------------------------------------------------------------------
template< typename T >
int randInt( 	T& rng,
				int mn,
				int mx )
{
	return int(rng() * (mx-mn) + mn);
}

//-------------------------------------------------------------------------------------------------
template< typename T >
float randFloat( 	T& rng,
					float mn,
					float mx )
{
	return rng() * (mx-mn) + mn;
}

//-------------------------------------------------------------------------------------------------
template< typename T >
Imath::V3f randDir( T& rng )
{
	Imath::V3f result;

	// random point inside a sphere
	while(true)
	{
		result = Imath::V3f( randFloat( rng, -1.0f, 1.0f )
		                   , randFloat( rng, -1.0f, 1.0f )
		                   , randFloat( rng, -1.0f, 1.0f ) );
		if( result.length() < 1.0f )
			break;
	}
	// then normalize
	result.normalize();

	return result;
}

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
