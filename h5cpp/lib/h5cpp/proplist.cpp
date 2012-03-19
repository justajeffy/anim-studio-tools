#include "proplist.hpp"

namespace h5cpp { namespace proplist {


// this is necessary because H5P_FILE_CREATE etc (see proplist_class.inc) are not constexprs
hid_t get_proplist_class_id(proplist_class cl)
{
	switch(cl)
	{
		#define _H5CPP_DEFN_INC(Enum) case _##Enum : return Enum;
		#include "inc/proplist_class.inc"
		#undef _H5CPP_DEFN_ENUM
	}
	return -1;
}


shared_hid_proplist create(proplist_class cl)
{
	hid_t proplist_id;
	hid_t class_id = get_proplist_class_id(cl);
	H5CPP_ERR_ON_NEG(proplist_id = H5Pcreate(class_id));
	return shared_hid_proplist(hid_proplist(proplist_id));
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
