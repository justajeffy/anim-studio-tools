#ifndef _NAPALM_TYPELABELS__H_
#define _NAPALM_TYPELABELS__H_

#include "util/type_info.hpp"
#include "typedefs.h"


/*
 * Compile-time type->label associations. Napalm represents the types it knows about with
 * convenience labels - for example, TypedBuffer<int> is represented as "IntBuffer". This
 * is NOT used as a form of RTTI - Napalm uses native RTTI for that. Instead, they are used
 * to give types more human-readable names in printouts, and for naming python bindings.
 */

#define _NAPALM_DEFINE_TYPELABEL3(T, Label) \
	template<> struct type_label<T> { static std::string value() { return #Label; } };

#define _NAPALM_DEFINE_TYPELABEL2(T, Label) 							\
	_NAPALM_DEFINE_TYPELABEL3(T, 					Label)				\
	_NAPALM_DEFINE_TYPELABEL3(TypedAttribute<T>, 	Label##Attrib)		\
	_NAPALM_DEFINE_TYPELABEL3(TypedBuffer<T>, 		Label##Buffer)		\
	_NAPALM_DEFINE_TYPELABEL3(BufferStoreCpu<T>, 	Label##CpuStore)

#define _NAPALM_DEFINE_TYPELABEL(T, Label)								\
	_NAPALM_DEFINE_TYPELABEL2(T, Label) 								\
	_NAPALM_DEFINE_TYPELABEL2(counted_vector<T>::type, Label##Vec) 		\
	_NAPALM_DEFINE_TYPELABEL2(counted_set<T>::type, Label##Set)


namespace napalm {

	template<typename T>
	struct type_label
	{
		static std::string value() {
			return util::get_type_name<T>();
		}
	};

	#define _NAPALM_TYPE_OP(T, Label) _NAPALM_DEFINE_TYPELABEL(T, Label)
	#include "types/all.inc"
	#undef _NAPALM_TYPE_OP

	// (aj)
	//template<> struct type_label<bool>					{ static std::string value(){ return "Bool"; } };
	//template<> struct type_label<TypedAttribute<bool> >	{ static std::string value(){ return "BoolAttrib"; } };

	template<> struct type_label<ObjectTable>{ static std::string value(){ return "ObjectTable"; } };
	template<> struct type_label<AttribTable>{ static std::string value(){ return "AttribTable"; } };

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
