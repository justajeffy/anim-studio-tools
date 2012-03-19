#include "datatype.hpp"


namespace h5cpp {

compound_datatype_base::compound_datatype_base(std::size_t size)
{
	hid_t dtype_id = H5Tcreate(H5T_COMPOUND, size);
	hid_datatype dtype_hid(dtype_id);
	shared_hid_datatype dtype(dtype_hid);
	m_dtypes.push_back(dtype);
}


compound_datatype_base::compound_datatype_base(const compound_datatype_base& rhs)
:	m_dtypes(rhs.m_dtypes)
{
}


compound_datatype_base::~compound_datatype_base()
{
	// destroy h5 datatype handles in the same order they were created
	while(!m_dtypes.empty())
		m_dtypes.pop_front();
}


namespace detail {

shared_hid_datatype _cdtype_inserter<char*>::insert(hid_t cdtype_id, std::size_t varoffset,
	const std::string& name)
{
	shared_hid_datatype dtype = datatype::get_type<vl_string>();
	H5CPP_ERR_ON_NEG(H5Tinsert(cdtype_id, name.c_str(), varoffset, dtype.id()));
	return dtype;
}

} // detail ns


namespace datatype {

compound_datatype_manager::map_type compound_datatype_manager::m_compound_creator_fns;


shared_hid_datatype get_fl_string_type(std::size_t nchars)
{
	if(nchars == 0)
		H5CPP_THROW("get_fl_string_type: nchars must be > 0");

	hid_t type_id = H5CPP_ERR_ON_NEG(H5Tcopy(H5T_C_S1));
	H5CPP_ERR_ON_NEG(H5Tset_size(type_id, nchars));
	return shared_hid_datatype(hid_datatype(type_id));
}


// static globals for simple data types
#define _H5CPP_DEFN_INC(Type, H5Type) \
	shared_hid_datatype g_h5cpp_##H5Type(hid_datatype(H5Type));
#include "inc/native_types.inc"
#undef _H5CPP_DEFN_INC


// get_type() specializations
#define _H5CPP_DEFN_INC(Type, H5Type) \
	template<> shared_hid_datatype get_type<Type>() { return g_h5cpp_##H5Type; }
#include "inc/native_types.inc"
#undef _H5CPP_DEFN_INC

template<>
shared_hid_datatype get_type<vl_string>()
{
	hid_t type_id = H5CPP_ERR_ON_NEG(H5Tcopy(H5T_C_S1));
	H5CPP_ERR_ON_NEG(H5Tset_size(type_id, H5T_VARIABLE));
	return shared_hid_datatype(hid_datatype(type_id));
}


// get_type(T&) specializations
template<>
shared_hid_datatype get_type<fl_string>(const fl_string& t)
{
	return get_fl_string_type(t.t.length() + 1);
}


} }











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
