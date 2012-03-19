#ifndef _H5CPP_ATTRIBUTE__H_
#define _H5CPP_ATTRIBUTE__H_

#include "hid_adaptor.hpp"
#include "space.hpp"
#include "datatype.hpp"


namespace h5cpp { namespace attrib {


	/*
	 * @brief write
	 * See H5Awrite
	 * Writes a value to an existing attribute. Note that a fixed-length string can be written
	 * to a variable-length string, and vice versa - the attribute type will not be changed,
	 * but the written string will be truncated/resized accordingly.
	 */
	template<typename T>
	void write(const hid_attribute_adaptor& attrib, const T& value)
	{
		shared_hid_datatype dtype = datatype::get_type(value);
		H5CPP_ERR_ON_NEG(H5Awrite(attrib.id(), dtype.id(), static_cast<const void*>(&value)));
	}

	void write(const hid_attribute_adaptor& attrib, const fl_string& value);
	void write(const hid_attribute_adaptor& attrib, const vl_string& value);
	void write(const hid_attribute_adaptor& attrib, const std::string& value);


	/*
	 * @brief create
	 * See H5Acreate, H5Awrite
	 */
	template<typename T>
	shared_hid_attribute create(const hid_location_adaptor& loc, const std::string& name, const T& value)
	{
		shared_hid_datatype dtype = datatype::get_type(value);
		shared_hid_dataspace space = space::create_scalar();
		hid_t attr_id = H5CPP_ERR_ON_NEG(
			H5Acreate(loc.id(), name.c_str(), dtype.id(), space.id(), H5P_DEFAULT, H5P_DEFAULT));

		hid_attribute attr(attr_id);
		write(attr, value);

		return shared_hid_attribute(hid_attribute(attr_id));
	}


	/*
	 * @brief get_type
	 * See H5Aget_type
	 */
	shared_hid_datatype get_type(const hid_attribute_adaptor& attr);

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
