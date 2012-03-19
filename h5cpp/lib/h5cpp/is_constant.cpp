#include "is_constant.h"
#include "error.hpp"
#include <hdf5/hdf5.h>
#include <set>
#include <map>


namespace h5cpp
{


bool is_constant(hid_t id)
{
	if(id == -1)
		return false;

	// because some constant values are zero (eg H5S_ALL, H5P_DEFAULT), and we don't have
	// type info, we're forced to assume that a zero-value id is a constant id :(
	if(id == 0)
		return true;

	H5I_type_t type = H5Iget_type(id);
	if(type == H5I_BADID)
		H5CPP_THROW("internal error: H5Iget_type failure in is_constant");

	return is_constant(type, id);
}


bool is_constant(H5I_type_t type, hid_t id)
{
	typedef std::set<hid_t> hid_set;

	if(id == -1)
		return false;

	switch(type)
	{
	case H5I_DATATYPE:
	{
		static bool init = false;
		static hid_set constant_ids;
		if(!init)
		{
			#define _H5CPP_DEFN_INC(Type, H5Type) constant_ids.insert(H5Type);
			#include "inc/native_types.inc"
			#undef _H5CPP_DEFN_INC
			init = true;
		}

		return (constant_ids.find(id) != constant_ids.end());
	}
	break;

	case H5I_DATASPACE:
	{
		return (id == H5S_ALL);
	}
	break;

	default:
		return false;
	}

	return false;
}


}


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
