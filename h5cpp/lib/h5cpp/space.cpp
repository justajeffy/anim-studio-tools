#include "space.hpp"
#include <sstream>


namespace h5cpp { namespace space {


dimvec create_dimvec(hsize_t dim1, hsize_t dim2, hsize_t dim3)
{
	dimvec dims(1, dim1);
	if(dim2 > 0)
	{
		dims.push_back(dim2);
		if(dim3 > 0)
			dims.push_back(dim3);
	}

	return dims;
}


std::size_t dimvec_nelems(const dimvec& dims)
{
	if(dims.empty())
		return 0;

	std::size_t nelems(1);
	for(dimvec::const_iterator it=dims.begin(); it!=dims.end(); ++it)
		nelems *= *it;

	return nelems;
}


std::string dimvec_as_string(const dimvec& dims)
{
	std::ostringstream strm;
	strm << '[';
	bool comma = false;
	for(dimvec::const_iterator it=dims.begin(); it!=dims.end(); ++it, comma=true)
	{
		if(comma)
			strm << ',';
		strm << *it;
	}
	strm << ']';
	return strm.str();
}


shared_hid_dataspace create_scalar()
{
	hid_t space_id = H5CPP_ERR_ON_NEG(H5Screate(H5S_SCALAR));
	return shared_hid_dataspace(hid_dataspace(space_id));
}


shared_hid_dataspace create_simple(const dimvec& dims)
{
	hid_t space_id = (dims.empty())?
		H5CPP_ERR_ON_NEG(H5Screate(H5S_NULL)) :
		H5CPP_ERR_ON_NEG(H5Screate_simple(dims.size(), &dims[0], NULL));

	return shared_hid_dataspace(hid_dataspace(space_id));
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
