#ifndef _NAPALM_TYPES__TYPES__H_
#define _NAPALM_TYPES__TYPES__H_

/*
 * Note: types are split over files in this directory in order to decrease compilation
 * time via parallel compilation (ala make -j)
 */

#include "../BufferStoreCpu.h"
#include "../TypedAttribute.h"
#include "../TypedBuffer.h"
#include "../Table.h"
#include "../typedefs.h"
#include "../typelabels.h"
#include "../Dispatcher.h"
#include "../util/counted_allocator.hpp"

// here we must include *all* boost archive types being used
#include "../archives.h"

#include <boost/serialization/string.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>

// this must be included at this point, ie it must appear *after* the
// previous includes... do not change the ordering
#include <boost/serialization/export.hpp>

#define _EXPORT_TYPE( T, Label ) \
		::boost::serialization::singleton< \
			::boost::archive::detail::guid_initializer<T> \
				>::get_mutable_instance().export_guid(#Label);

#define _EXPORT_TYPE_CATEGORY( T, Label ) \
		_EXPORT_TYPE( napalm::TypedAttribute<T>, Label##Attrib )\
		_EXPORT_TYPE( napalm::TypedBuffer<T>, Label##Buffer )\
		_EXPORT_TYPE( napalm::BufferStoreCpu<T>, Label##CpuStore )

#define _EXPORT_TYPES(T, Label) \
	namespace napalm{ \
	void export_type_##Label() \
	{\
	_EXPORT_TYPE_CATEGORY( T, Label ) \
	_EXPORT_TYPE_CATEGORY( napalm::counted_vector<T>::type, Label##Vec ) \
	_EXPORT_TYPE_CATEGORY( napalm::counted_set<T>::type, Label##Set )\
	} }

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
