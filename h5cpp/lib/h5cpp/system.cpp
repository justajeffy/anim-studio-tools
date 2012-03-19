#include "system.h"
#include "datatype.hpp"
#include <cstdlib>

namespace h5cpp {


detail::scoped_suppress_hdf5_errors system::m_errorscope;


void system::init()
{
	static bool initialised = false;
	if(initialised)
		return;

	H5open();
	atexit(close_h5cpp);
	initialised = true;
}


void system::close()
{
}


void close_h5cpp()
{
	system::close();
}


// force initialisation
struct __init_h5cpp
{
	__init_h5cpp() { system::init(); }
};


static __init_h5cpp __init_h5cpp_inst;


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
