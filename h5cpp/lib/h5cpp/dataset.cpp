#include "dataset.hpp"


namespace h5cpp { namespace dataset {


shared_hid_dataset create(const hid_location_adaptor& loc, const std::string& name,
	const hid_datatype_adaptor& dtype, const hid_dataspace_adaptor& space)
{
	hid_t dset_id = H5CPP_ERR_ON_NEG(H5Dcreate(loc.id(), name.c_str(), dtype.id(),
		space.id(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
	return shared_hid_dataset(hid_dataset(dset_id));
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
