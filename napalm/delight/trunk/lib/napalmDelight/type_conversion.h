#ifndef _NAPALM_DELIGHT_TYPE_CONVERSION__H_
#define _NAPALM_DELIGHT_TYPE_CONVERSION__H_

#include <ri.h>
#include <rx.h>

namespace napalm_delight
{

//----------------------------------------------------------------------------------------------------------------------
inline void convert(	const Imath::M44f &src,
						RtMatrix dst )
{
	for ( size_t i = 0 ; i < 4 ; i++ )
		for ( size_t j = 0 ; j < 4 ; j++ )
			dst[ i ][ j ] = src[ i ][ j ];
}

//----------------------------------------------------------------------------------------------------------------------
inline void convert(	const RtMatrix src,
                    	Imath::M44f& dst )
{
	for ( size_t i = 0 ; i < 4 ; i++ )
		for ( size_t j = 0 ; j < 4 ; j++ )
			dst[ i ][ j ] = src[ i ][ j ];
}

//----------------------------------------------------------------------------------------------------------------------
inline void convert(	const Imath::Box3f &src,
						RtBound dst )
{
	dst[ 0 ] = src.min.x;
	dst[ 1 ] = src.max.x;
	dst[ 2 ] = src.min.y;
	dst[ 3 ] = src.max.y;
	dst[ 4 ] = src.min.z;
	dst[ 5 ] = src.max.z;
}

//----------------------------------------------------------------------------------------------------------------------
inline void convert(	const RtBound src,
						Imath::Box3f &dst )
{
	dst.min.x = src[ 0 ];
	dst.max.x = src[ 1 ];
	dst.min.y = src[ 2 ];
	dst.max.y = src[ 3 ];
	dst.min.z = src[ 4 ];
	dst.max.z = src[ 5 ];
}

//----------------------------------------------------------------------------------------------------------------------
// pad an RtBound
inline void pad( RtBound bound, float val )
{
	bound[0] -= val;
	bound[1] += val;
	bound[2] -= val;
	bound[3] += val;
	bound[4] -= val;
	bound[5] += val;
}

//----------------------------------------------------------------------------------------------------------------------
// pad an Imath::Box3f
inline void pad( Imath::Box3f& bound, float val )
{
	Imath::V3f delta( val );
	bound.min -= delta;
	bound.max += delta;
}


//-------------------------------------------------------------------------------------------------
template< typename SIMPLE_TYPE >
bool getRxAttribute( const std::string& name, SIMPLE_TYPE& dst )
{
	RxInfoType_t o_resultType;
	int o_resultCount = 0;
	bool success = (RIE_NOERROR == RxAttribute( name.c_str(), &dst, sizeof( dst ), &o_resultType, &o_resultCount ) );
	return success;
}

//-------------------------------------------------------------------------------------------------
template< typename SIMPLE_TYPE >
bool getRxOption( const std::string& name, SIMPLE_TYPE& dst )
{
	RxInfoType_t o_resultType;
	int o_resultCount = 0;
	bool success = (RIE_NOERROR == RxOption( name.c_str(), &dst, sizeof( dst ), &o_resultType, &o_resultCount ) );
	return success;
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
