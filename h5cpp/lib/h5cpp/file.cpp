#include "file.hpp"
#include "proplist.hpp"


namespace h5cpp { namespace file {


void file_access_props::apply(hid_proplist proplist) const
{
	if(m_fclose_degree_set)		H5CPP_ERR_ON_NEG(H5Pset_fclose_degree(proplist.id(), m_fclose_degree));
}


shared_hid_file open(const std::string& filepath, bool readonly, const file_access_props& aprops)
{
	hid_t file_id;
	shared_hid_proplist pl(proplist::create(proplist::_H5P_FILE_ACCESS));
	aprops.apply(pl.hid());

	H5CPP_ERR_ON_NEG(file_id = H5Fopen(filepath.c_str(),
		(readonly)? H5F_ACC_RDONLY : H5F_ACC_RDWR, pl.id()));

	return shared_hid_file(hid_file(file_id));
}


shared_hid_file create(const std::string& filepath, const file_access_props& aprops)
{
	hid_t file_id;
	shared_hid_proplist pl(proplist::create(proplist::_H5P_FILE_ACCESS));
	aprops.apply(pl.hid());

	H5CPP_ERR_ON_NEG(file_id = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, pl.id()));
	return shared_hid_file(hid_file(file_id));
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
