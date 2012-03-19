#ifndef _H5CPP_IS_CONSTANT__H_
#define _H5CPP_IS_CONSTANT__H_

#include <hdf5/H5Ipublic.h>
#include <hdf5/H5Tpublic.h>

/*
 * The functions here are necessary because hdf5 is inconsistent in how it deals with handles.
 * There are many 'constant' handles available for use in hdf5, that can be passed to API
 * calls interchangeably with hdf5 ids that have been generated from creating/opening resouces
 * (such as files, groups etc). For example:
 *
 * H5T_NATIVE_XXX are all constant H5I_DATATYPE ids, yet H5Iis_valid() will fail on these.
 *
 * H5S_ALL is treated as an H5I_DATASPACE by the API, yet both H5Iis_valid() and H5Iget_type()
 * fail on it.
 *
 * The ability to safely tell if an id is a constant allows h5cpp to treat them consistently,
 * and avoid reporting them as invalid, or attempting to close() them.
 */

namespace h5cpp
{

	/*
	 * @brief is_constant
	 * Returns true if the given id is an hdf5 constant (eg H5S_ALL, H5T_NATIVE_XXX). Returns
	 * false if id is not a constant, or is bad.
	 */
	bool is_constant(hid_t id);


	/*
	 * @brief is_constant
	 * Returns true if the given id is an hdf5 constant (eg H5S_ALL, H5T_NATIVE_XXX). Returns
	 * false if id is not a constant, or is bad.
	 */
	bool is_constant(H5I_type_t type, hid_t id);

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
