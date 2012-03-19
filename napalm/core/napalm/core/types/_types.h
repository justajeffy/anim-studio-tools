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


#define _INIT_NAPALM_TYPE(T, Label) \
	detail::initialise_napalm_type<T>(#Label);


#define _INSTANTIATE_NAPALM_TYPE1(T, Label) 									\
	template class TypedAttribute<T>;											\
	template class TypedBuffer<T>;												\
	template class BufferStoreCpu<T>;											\
	void __initialise_napalm_base_type_##Label()								\
	{																			\
		_INIT_NAPALM_TYPE(TypedAttribute<T>, 				Label##Attrib);		\
		_INIT_NAPALM_TYPE(TypedBuffer<T>, 					Label##Buffer);		\
		_INIT_NAPALM_TYPE(BufferStoreCpu<T>, 				Label##CpuStore);	\
		detail::initialise_napalm_base_type_primitive<T>(#Label);				\
	}

#define _INSTANTIATE_NAPALM_TYPE2(T, Label) 									\
	template class TypedAttribute<T>;											\
	template class TypedBuffer<T>;												\
	template class BufferStoreCpu<T>;											\
	void __initialise_napalm_base_type_##Label()								\
	{																			\
		_INIT_NAPALM_TYPE(TypedAttribute<T>, 				Label##Attrib);		\
		_INIT_NAPALM_TYPE(TypedBuffer<T>, 					Label##Buffer);		\
		_INIT_NAPALM_TYPE(BufferStoreCpu<T>, 				Label##CpuStore);	\
		detail::initialise_napalm_base_type<T>(#Label);							\
	}


#define _INSTANTIATE_NAPALM_TYPE(T, Label)										\
namespace napalm {																\
	_INSTANTIATE_NAPALM_TYPE1(T, Label)											\
	_INSTANTIATE_NAPALM_TYPE2(counted_vector<T>::type, Label##Vec)				\
	_INSTANTIATE_NAPALM_TYPE2(counted_set<T>::type, Label##Set)					\
	void _initialise_napalm_base_type_##Label() 								\
	{																			\
		__initialise_napalm_base_type_##Label();								\
		__initialise_napalm_base_type_##Label##Vec();							\
		__initialise_napalm_base_type_##Label##Set();							\
	}																			\
}

namespace napalm { namespace detail {

	template<typename T>
	void initialise_napalm_type(const char* label)
	{
		// register with dispatcher
		Dispatcher::entry e;
		e.m_typeLabel = type_label<T>::value();
		Dispatcher::instance().setTypeEntry<T>(e);
	}

	template<typename T>
	void initialise_napalm_base_type(const char* label)
	{
		// register with dispatcher
		Dispatcher::base_entry e;
		e.m_get_saveable_store_fn = detail::getSaveableStore<T>;
		Dispatcher::instance().AddSelfOnlyMapRow<T>();
		Dispatcher::instance().setBaseTypeEntry<T>(e);
	}

	// 'primitive' means any type which you can have sets or vectors of
	template<typename T>
	void initialise_napalm_base_type_primitive(const char* label)
	{
		// register with dispatcher
		Dispatcher::base_entry e;
		e.m_get_saveable_store_fn = detail::getSaveableStore<T>;
		Dispatcher::instance().AddMapRow<T>();
		Dispatcher::instance().setBaseTypeEntry<T>(e);
	}

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
