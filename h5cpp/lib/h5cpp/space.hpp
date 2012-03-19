#ifndef _H5CPP_SPACE__H_
#define _H5CPP_SPACE__H_

#include "hid_wrap.hpp"
#include <vector>


namespace h5cpp {

	typedef std::vector<hsize_t> dimvec;

namespace space {

	/*
	 * @brief create_dims
	 * Create an std::vector describing the dimensions of a dataspace
	 */
	dimvec create_dimvec(hsize_t dim1, hsize_t dim2=0, hsize_t dim3=0);


	/*
	 * @brief dims_size
	 * Given an std::vector describing the dimensions of a dataspace, return the total
	 * number of elements in the dataspace.
	 */
	std::size_t dimvec_nelems(const dimvec& dims);


	/*
	 * @brief dimvec_as_string
	 * Return a string representation of a dimvec
	 */
	std::string dimvec_as_string(const dimvec& dims);


	/*
	 * @brief create_scalar
	 * See H5Screate
	 */
	shared_hid_dataspace create_scalar();


	/*
	 * @brief create_simple
	 * See H5Screate_simple
	 * @param dims Dataspace dimensions. If empty, a null dataset will be created.
	 */
	shared_hid_dataspace create_simple(const dimvec& dims);

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
