#ifndef _H5CPP_CLOSE__H_
#define _H5CPP_CLOSE__H_

#include "hid_types.hpp"


namespace h5cpp
{

	namespace detail
	{
		inline herr_t close(const hid_file& hid) 		{ return H5Fclose(hid.id()); }
		inline herr_t close(const hid_group& hid) 		{ return H5Gclose(hid.id()); }
		inline herr_t close(const hid_datatype& hid) 	{ return H5Tclose(hid.id()); }
		inline herr_t close(const hid_dataspace& hid) 	{ return H5Sclose(hid.id()); }
		inline herr_t close(const hid_dataset& hid) 	{ return H5Dclose(hid.id()); }
		inline herr_t close(const hid_attribute& hid)	{ return H5Aclose(hid.id()); }
		inline herr_t close(const hid_proplist& hid)	{ return H5Pclose(hid.id()); }


	}


	/*
	 * @brief close
	 * Closes an hdf5 id
	 */
	void close(hid_t id);


	/*
	 * @brief close
	 * Closes an hdf5 id, assuming the specified type
	 */
	void close(hid_t id, H5I_type_t type);


	/*
	 * @brief close
	 * Closes an hid_object
	 */
	void close(const hid_object& hid);


	/*
	 * @brief close
	 * Closes a strongly-typed hid
	 */
	template<typename HID>
	void close(const HID& hid)
	{
		if((!hid) || (is_constant(HID::s_type_i, hid.id())))
			return;

		H5CPP_ERR_ON_NEG(detail::close(hid));
	}


	/*
	 * @brief close
	 * Closes an hid_variant
	 */
	template<typename Sequence>
	void close(const hid_variant<Sequence>& hid)
	{
		close(hid.id(), hid.type());
	}

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
