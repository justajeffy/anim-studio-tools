/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: random.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_random_h
#define grind_random_h

//! @cond DEV

//-------------------------------------------------------------------------------------------------
#include "device_vector.h"
//#include <drd/singleton.h>
#include "singleton_rebuildable.h"

namespace grind
{

//-------------------------------------------------------------------------------------------------
//! uniform distribution float random number 0->1
float nRand();

//-------------------------------------------------------------------------------------------------
//! uniform distribution random number centred around 1 with a user defined deviation
float mRand( float dev );

//-------------------------------------------------------------------------------------------------
//! pick a number between 0 and max_value-1
int iRand( int max_value );

//-------------------------------------------------------------------------------------------------
//! get an offset into the random number table
size_t randSeedHash( const char* str );

//-------------------------------------------------------------------------------------------------
struct DeviceRandom
: public drd::RebuildableSingleton< DeviceRandom >
{
	friend class drd::RebuildableSingleton< DeviceRandom >;

	//! access a table of normalized random numbers
	const DeviceVector< float >& getNormRandTable() const
	{
		return m_NormRandTable;
	}

	//! access a table of pairs of random numbers that fall within a unit disc
	const DeviceVector< Imath::V2f >& getDiscRandTable() const
	{
		return m_DiscRandTable;
	}

	~DeviceRandom();

private:
	//! construct and setup table
	DeviceRandom();

	void constructNormDistributionTable();
	void constructDiscDistributionTable();

	//! random table on device
	DeviceVector< float > m_NormRandTable;

	//! random table on device
	DeviceVector< Imath::V2f > m_DiscRandTable;
};

} // namespace grind

//! @endcond

#endif /* grind_random_h */


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
