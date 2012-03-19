/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: host_vector.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_host_vector_h
#define grind_host_vector_h

//! @cond DEV

//-------------------------------------------------------------------------------------------------
#include <vector>
#include "log.h"

namespace grind {

template< typename T >
class DeviceVector;

//-------------------------------------------------------------------------------------------------
//! a vector of data located on the host
template< typename T >
class HostVector
: public std::vector<T>
{
public:
	HostVector()
	: std::vector<T>()
	{}

	HostVector( size_t sz )
	: std::vector<T>( sz )
	{}

	HostVector( size_t sz, const T& iv )
	: std::vector<T>( sz, iv )
	{}

	//! dump to log
	void dump() const;

	void setValue( const DeviceVector<T>& src )
	{
		// let DeviceVector do the work
		src.getValue( *this );
	}
};

template< typename T >
void HostVector< T >::dump() const
{
	for ( int i = 0 ; i < this->size() ; ++i )
	{
		std::cerr << this->operator[]( i ) << ", ";
	}
	std::cerr << "\n";
}


} // namespace grind

//! @endcond

#endif /* grind_host_vector_h */


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
