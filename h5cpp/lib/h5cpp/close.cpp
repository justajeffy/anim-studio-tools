#include "close.hpp"
#include "system.h"


namespace h5cpp
{


void _close(hid_t id, H5I_type_t type)
{
	herr_t status = -1;
	switch(type)
	{
	case H5I_FILE:			status = H5Fclose(id); break;
	case H5I_GROUP:			status = H5Gclose(id); break;
	case H5I_DATATYPE:		status = H5Tclose(id); break;
	case H5I_DATASPACE:		status = H5Sclose(id); break;
	case H5I_DATASET:		status = H5Dclose(id); break;
	case H5I_ATTR:			status = H5Aclose(id); break;
	case H5I_GENPROP_LST:	status = H5Pclose(id); break;
	default:
		H5CPP_THROW("internal error: don't know how to close object type " << type);
	}

	if(status < 0)
		H5CPP_THROW("close failed with status " << status << " on type #" << type);
}


void close(hid_t id, H5I_type_t type)
{
	if((id == -1) || (is_constant(type, id)))
		return;

	if(!H5Iis_valid(id))
		H5CPP_THROW("attempted to close bad hdf5 id: " << id);

	_close(id, type);
}


void close(hid_t id)
{
	if((id == -1) || (is_constant(id)))
		return;

	if(!H5Iis_valid(id))
		H5CPP_THROW("attempted to close bad hdf5 id: " << id);

	H5I_type_t t = H5CPP_ERR_ON_NEG(H5Iget_type(id));
	_close(id, t);
}


void close(const hid_object& hid)
{
	close(hid.id());
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
