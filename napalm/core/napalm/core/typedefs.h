#ifndef _NAPALM_TYPEDEFS__H_
#define _NAPALM_TYPEDEFS__H_

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <map>
#include <set>
#include <string>
#include "util/counted_allocator.hpp"
#include "util/less_than.hpp"
#include "system.h"
#include "types/type_enable.h"


/*
 * forward declarations
 */

namespace napalm {

	class Object;
	class Attribute;
	class Buffer;
	class BufferStore;
	class BufferStoreHolder;

	template<typename T> 		class TypedBufferStore;
	template<typename T> 		class BufferStoreCpu;
	template<typename T> 		class TypedAttribute;
	template<typename T> 		class TypedBuffer;
	template<typename T> 		class Wrapper;
	template<typename Value>	class Table;
	template<typename Value>	class List;

	template<typename T>
	struct counted_vector {
		typedef std::vector<T, util::counted_allocator<T> > type;
	};

	template<typename T>
	struct counted_set {
		typedef std::set<T, util::less<T>, util::counted_allocator<T> > type;
	};

}

class half;

namespace Imath {

	template<typename T> class Vec2;
	template<typename T> class Vec3;
	template<typename T> class Vec4;
	template<typename T> class Matrix33;
	template<typename T> class Matrix44;
	template<typename T> class Box;

}


/*
 * typedefs
 */

#define _NAPALM_DEFINE_TYPE2(T, Label)																\
	typedef TypedAttribute<T> 												Label##Attrib;			\
	typedef TypedBuffer<T> 													Label##Buffer;			\
	typedef TypedBufferStore<T> 											Label##BufferStore;		\
	typedef BufferStoreCpu<T> 												Label##CpuStore;		\
	typedef boost::shared_ptr<TypedAttribute<T> > 							Label##AttribPtr;		\
	typedef boost::shared_ptr<const TypedAttribute<T> > 					Label##AttribCPtr;		\
	typedef boost::shared_ptr<TypedBuffer<T> > 								Label##BufferPtr;		\
	typedef boost::shared_ptr<const TypedBuffer<T> > 						Label##BufferCPtr;		\
	typedef boost::shared_ptr<BufferStoreCpu<T> > 							Label##CpuStorePtr;		\
	typedef boost::shared_ptr<const BufferStoreCpu<T> > 					Label##CpuStoreCPtr;

#define _NAPALM_DEFINE_TYPE(T, Label) 							\
	_NAPALM_DEFINE_TYPE2(T, Label) 								\
	_NAPALM_DEFINE_TYPE2(counted_vector<T>::type, Label##Vec) 	\
	_NAPALM_DEFINE_TYPE2(counted_set<T>::type, Label##Set)


namespace napalm {

	#define _NAPALM_TYPE_OP(T, Label) _NAPALM_DEFINE_TYPE(T, Label)
	#include "types/all.inc"
	#undef _NAPALM_TYPE_OP

// (aj) remove
	typedef Wrapper<Attribute>									TableKey;

	typedef std::vector<TableKey>								table_key_vector;

	typedef Table<Object>										ObjectTable;
	typedef Table<Attribute>									AttribTable;

	// (aj) change to match other typedefs
	typedef boost::shared_ptr<Object> 							object_ptr;
	typedef boost::shared_ptr<const Object> 					c_object_ptr;
	typedef std::set<const Object*> 							object_rawptr_set;
	typedef std::map<const Object*, object_ptr>					object_clone_map;

	typedef boost::shared_ptr<Attribute> 						attrib_ptr;
	typedef boost::shared_ptr<const Attribute> 					c_attrib_ptr;

	typedef boost::shared_ptr<Buffer>							buffer_ptr;
	typedef boost::shared_ptr<const Buffer>						c_buffer_ptr;

	typedef boost::shared_ptr<BufferStore> 						store_ptr;
	typedef boost::shared_ptr<const BufferStore> 				c_store_ptr;

	typedef boost::shared_ptr<BufferStoreHolder> 				store_holder_ptr;
	typedef boost::shared_ptr<const BufferStoreHolder> 			c_store_holder_ptr;
	typedef boost::weak_ptr<BufferStoreHolder>					store_holder_wptr;

	typedef boost::shared_ptr<ObjectTable> 						object_table_ptr;
	typedef boost::shared_ptr<const ObjectTable> 				c_object_table_ptr;

	typedef boost::shared_ptr<AttribTable> 						attrib_table_ptr;
	typedef boost::shared_ptr<const AttribTable> 				c_attrib_table_ptr;

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
