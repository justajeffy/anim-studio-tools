/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: hash_search.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_hash_search_h
#define grind_hash_search_h

//-------------------------------------------------------------------------------------------------
#include "device_vector.h"


namespace grind
{

// different orders of magnitude of primes
#define PRIME_100		101
#define PRIME_1000  	1009
#define PRIME_10000 	10007
#define PRIME_100000	100003
#define PRIME_1000000	999983

//-------------------------------------------------------------------------------------------------
/*! \brief Searcher based on a hash table.
 * Refer to...
 * - http://www.iro.umontreal.ca/labs/infographie/papers/Clavet-2005-PVFS/pvfs.pdf
 * - http://developer.download.nvidia.com/presentations/2008/GDC/GDC08_ParticleFluids.pdf etc
 */
class HashSearch
{

public:

	//! make a searcher
	HashSearch( Imath::V3f i_CellDim,
				int i_TableSize = PRIME_10000 );

	//! build the tables based on supplied points
	void build( const DeviceVector< Imath::V3f >& i_P //!< data set
	            );

	//! vectorized search returning indices of closest points ( -1 = not found )
	void findClosest(	const DeviceVector< Imath::V3f >& i_P,		//!< data set
						const DeviceVector< Imath::V3f >& i_QueryP,	//!< query locations
						DeviceVector< int >& o_ResultIndices	//!< resulting indices of closest points
					);

	//! cpu based O(N*M) search (useful for verifying results)
	void findClosestGold(	const DeviceVector< Imath::V3f >& i_P,		//!< data set
							const DeviceVector< Imath::V3f >& i_QueryP,	//!< query locations
							DeviceVector< int >& o_ResultIndices	//!< resulting indices of closest points
						);

	//! dump internal data
	void dump() const;

private:

	//! cell dimensions
	Imath::V3f m_CellDimension;

	//! size of hash table
	int m_TableSize;

	//! hash values for points
	DeviceVector< int > m_HashVal;

	//! table to index into points table
	DeviceVector< int > m_PIndices;

	//! start indices m_PIndices so searcher can track through subset of P very quickly
	DeviceVector< int > m_HashIndices;
};

//! test the hash search with some random values
bool testHashSearch();

}

#endif /* grind_hash_search_h */


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
