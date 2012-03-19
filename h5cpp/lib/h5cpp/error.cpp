#include "error.hpp"

namespace h5cpp { namespace detail {


struct hdf5_error_info {
	std::ostringstream m_msg;
};


herr_t hdf5_error_walk(unsigned int n, const H5E_error_t *err_desc, void *blind_data)
{
	hdf5_error_info* perr = static_cast<hdf5_error_info*>(blind_data);

	perr->m_msg << "\t#" << n << ": " << err_desc->file_name << " line " << err_desc->line
		<< " in " << err_desc->func_name << "()";

	if(err_desc->desc)
		perr->m_msg << ": " << err_desc->desc;

	perr->m_msg << '\n';
	return 0;
}


void throw_hdf5_error(const std::string& msg)
{
	hdf5_error_info err;
	err.m_msg << msg;
	H5Ewalk(H5E_DEFAULT, H5E_WALK_DOWNWARD, hdf5_error_walk, &err);
	throw hdf5_error(err.m_msg.str());
}


scoped_suppress_hdf5_errors::scoped_suppress_hdf5_errors()
{
	H5Eget_auto(H5E_DEFAULT, &m_prevFunc, &m_prevClientData);
	assert(m_prevFunc != NULL);
	H5Eset_auto(H5E_DEFAULT, NULL, NULL);
}


scoped_suppress_hdf5_errors::~scoped_suppress_hdf5_errors()
{
	H5Eset_auto(H5E_DEFAULT, m_prevFunc, m_prevClientData);
}


} } // ns

















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
