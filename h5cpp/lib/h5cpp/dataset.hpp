#ifndef _H5CPP_DATASET__H_
#define _H5CPP_DATASET__H_

#include "hid_adaptor.hpp"
#include "space.hpp"
#include "datatype.hpp"


namespace h5cpp { namespace dataset {

	/*
	 * @brief create
	 * See H5Dcreate
	 */
	shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name,
		const hid_datatype_adaptor& dtype, const hid_dataspace_adaptor& space);

	/*
	 * @brief create
	 * Create a dataset and write a single value into it. The dimensions of the resulting
	 * dataset will be [1].
	 */
	template<typename T>
	shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name,
		const T& data);

	/*
	 * @brief create
	 * Create a dataset and write a vector of data into it.
	 * @param data_dims Dimensions of data, defaults to [data.size()].
	 * @param space Dataspace of dataset, defaults to matching dimensions of data.
	 */
	template<typename T>
	shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name,
		const std::vector<T>& data, const dimvec& data_dims = dimvec(),
		const hid_dataspace_adaptor& space = hid_dataspace_adaptor());

	/*
	 * @brief create
	 * Create a dataset and write a raw array of data into it.
	 * @param data_dims Dimensions of data.
	 * @param space Dataspace of dataset, defaults to matching dimensions of data.
	 */
	template<typename T>
	shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name,
		const T* data, const dimvec& data_dims,
		const hid_dataspace_adaptor& space = hid_dataspace_adaptor());


///////////////////////// impl


template<typename T>
shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name, const T& data)
{
	return create(loc, name, &data, space::create_dimvec(1));
}


template<typename T>
shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name,
	const std::vector<T>& data, const dimvec& data_dims, const hid_dataspace_adaptor& space)
{
	if(data_dims.empty())
		return create(loc, name, &data[0], space::create_dimvec(data.size()), space);
	else
	{
		std::size_t dim_elems = space::dimvec_nelems(data_dims);
		if(data.size() != dim_elems)
			H5CPP_THROW("data element count mismatch in dataset::create (vector length="
				<< data.size() << ", data dims=" << space::dimvec_as_string(data_dims) << ")");

		return create(loc, name, &data[0], data_dims, space);
	}
}


template<typename T>
shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name,
	const T* data, const dimvec& data_dims, const hid_dataspace_adaptor& space)
{
	// create dataspaces
	shared_hid_dataspace memspace = space::create_simple(data_dims);
	const hid_dataspace& filespace = (space)? space.hid() : memspace.hid();

	// create dataset
	shared_hid_datatype dtype = datatype::get_type<T>();
	shared_hid_dataset dset = create(loc, name, dtype.hid(), filespace.id());

	// write data into dset
	if(space::dimvec_nelems(data_dims) > 0)
		H5CPP_ERR_ON_NEG(H5Dwrite(dset.id(), dtype.id(), memspace.id(), H5S_ALL, H5P_DEFAULT, data));

	return dset;
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
