/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: $"
 * SVN_META_ID = "$Id: ocl_manager.h 42544 2010-08-17 04:31:03Z allan.johns $"
 */

#ifndef grind_ocl_manager_h
#define grind_ocl_manager_h

//-------------------------------------------------------------------------------------------------
#include "grind/util/cl.hpp"
#include <boost/scoped_ptr.hpp>

//! @cond DEV

namespace grind
{

//-------------------------------------------------------------------------------------------------
//! a class that manages an OpenCL context
class OclManager
{
	//! the OpenCL context
	boost::scoped_ptr< cl::Context > m_Context;

	//! the list of devices
	std::vector< cl::Device > m_Devices;

	//! the command queue
	boost::scoped_ptr< cl::CommandQueue > m_Queue;

public:
	//! default constructor
	OclManager();

	static void initialize();

	//! destructor
	~OclManager();

	//! is the manager good to go?
	bool good();
};

//! One OclManager per grind instance
extern OclManager s_OCLManager;

} // grind

//! @endcond

#endif /* grind_ocl_manager */


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
