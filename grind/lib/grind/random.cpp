/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: random.cpp 45312 2010-09-09 06:11:02Z chris.cooper $"
 */

//-------------------------------------------------------------------------------------------------

#include <drdDebug/log.h>
DRD_MKLOGGER( L, "drd.grind.Random" );

#include "random.h"
#include "host_vector.h"

#include <stdlib.h>

//-------------------------------------------------------------------------------------------------
// prime table size
#define RANDOM_TABLE_SIZE 999983

//-------------------------------------------------------------------------------------------------
using namespace grind;
using namespace drd;

//-------------------------------------------------------------------------------------------------
float grind::nRand()
{
	return rand() / float( RAND_MAX );
}

//-------------------------------------------------------------------------------------------------
float grind::mRand( float dev )
{
	return 1.0f + ( grind::nRand() * 2.0f - 1.0f ) * dev;
}

//-------------------------------------------------------------------------------------------------
int grind::iRand( int max_value )
{
	return rand() % max_value;
}


//-------------------------------------------------------------------------------------------------
void DeviceRandom::constructNormDistributionTable()
{
	HostVector< float > host_table;
	host_table.reserve( RANDOM_TABLE_SIZE );

	srand( 0 );
	for ( int i = 0 ; i < RANDOM_TABLE_SIZE ; ++i )
	{
		host_table.push_back( nRand() );
	}
	m_NormRandTable.setValue( host_table );
}

//-------------------------------------------------------------------------------------------------
void DeviceRandom::constructDiscDistributionTable()
{
	HostVector< Imath::V2f > host_table( RANDOM_TABLE_SIZE );

	srand( 9883 );

	// setup tables (scraggle offsets etc)
	for ( int i = 0 ; i < RANDOM_TABLE_SIZE ; ++i )
	{
		float x, y;
		float dist = 2;

		while ( dist > 1 )
		{
			// random numbers -1 -> 1
			x = nRand() * 2.0f - 1.0f;
			y = nRand() * 2.0f - 1.0f;
			dist = x * x + y * y; // don't need sqrt as we're testing against radius 1
		}
		host_table[ i ].x = x;
		host_table[ i ].y = y;
	}

	m_DiscRandTable.setValue( host_table );

}

//-------------------------------------------------------------------------------------------------
DeviceRandom::DeviceRandom()
// note: disable OpenGL for these ones
: m_NormRandTable(0,0)
, m_DiscRandTable(0,0)
{
	DRD_LOG_DEBUG( L, "DeviceRandom constructor called from thread: " << pthread_self() );
	DRD_LOG_INFO( L, "constructing random number tables" );
	constructNormDistributionTable();
	constructDiscDistributionTable();
}

//-------------------------------------------------------------------------------------------------
DeviceRandom::~DeviceRandom()
{
	DRD_LOG_DEBUG( L, "DeviceRandom destructor called from thread: " << pthread_self() );
}

//-------------------------------------------------------------------------------------------------
// djb2 hash function http://www.cse.yorku.ca/~oz/hash.html
size_t
grind::randSeedHash( const char *str)
{
    size_t hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash % RANDOM_TABLE_SIZE; // modified by cc to return within range of random table size
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
