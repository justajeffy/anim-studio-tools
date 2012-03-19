#ifndef _H5CPP_SYSTEM__H_
#define _H5CPP_SYSTEM__H_

#include "error.hpp"
#include <set>
#include <map>


namespace h5cpp
{

	// TODO change this into a singleton, this means h5cpp will function correctly when
	// statically linked, right now it will not.
	class system
	{
	public:

		/*
		 * @brief init
		 * Initialize the h5cpp system.
		 */
		static void init();

		/*
		 * @brief close
		 * Shut down the h5cpp system.
		 */
		static void close();

	protected:

		static detail::scoped_suppress_hdf5_errors m_errorscope;
	};


extern "C" void close_h5cpp();

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
