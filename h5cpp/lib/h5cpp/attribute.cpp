#include "attribute.hpp"
#include "object.hpp"
#include <vector>

#include <iostream>


namespace h5cpp { namespace attrib {


void write(const hid_attribute_adaptor& attrib, const fl_string& value) {
	write(attrib, value.t);
}

void write(const hid_attribute_adaptor& attrib, const vl_string& value) {
	write(attrib, value.t);
}

void write(const hid_attribute_adaptor& attrib, const std::string& value)
{
	hid_t attr_type_id = H5CPP_ERR_ON_NEG(H5Aget_type(attrib.id()));
	if(H5Tget_class(attr_type_id) != H5T_STRING)
		H5CPP_THROW("Attempted to set string value on non-string attribute '"
			<< object::get_name(attrib.id()) << '\'');

	const char* cstr = value.c_str();

	if(H5Tis_variable_str(attr_type_id))
		H5CPP_ERR_ON_NEG(H5Awrite(attrib.id(), attr_type_id, &cstr));
	else
	{
		std::size_t attriblen = H5Aget_storage_size(attrib.id());
		if(attriblen > value.length())
		{
			// avoid reading past end of value into unalloc'd mem
			std::vector<char> buf(attriblen,'\0');
			std::copy(cstr, cstr+value.length(), &buf[0]);
			H5CPP_ERR_ON_NEG(H5Awrite(attrib.id(), attr_type_id, &buf[0]));
		}
		else
			H5CPP_ERR_ON_NEG(H5Awrite(attrib.id(), attr_type_id, cstr));
	}
}


shared_hid_datatype get_type(const hid_attribute_adaptor& attr)
{
	hid_t type_id = H5CPP_ERR_ON_NEG(H5Aget_type(attr.id()));
	return shared_hid_datatype(hid_datatype(type_id));
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
