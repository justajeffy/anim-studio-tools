/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: counting_iterator.h 42733 2010-08-18 06:12:21Z allan.johns $"
 */

#ifndef grind_counting_iterator_h
#define grind_counting_iterator_h

#include <thrust/iterator/counting_iterator.h>

namespace grind
{

#ifdef __DEVICE_EMULATION__
	//! a host counting iterator
	typedef thrust::counting_iterator< int, thrust::host_space_tag > CountingIterator;
#else
	//! a device counting iterator
	typedef thrust::counting_iterator< int, thrust::device_space_tag > CountingIterator;
#endif

} // grind namespace


#endif /* grind_counting_iterator_h */


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
