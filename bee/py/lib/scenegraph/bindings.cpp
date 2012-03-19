/*
 * python_bindings.cpp
 *
 *  Created on: Oct 9, 2009
 *      Author: stephane.bertout
 */


#include <boost/python.hpp>
using namespace boost::python;

#include <kernel/pythonHelper.h>
#include <scenegraph/camera.h>

using namespace bee;

void importScenegraphBindings()
{
	class_<Camera>( "Camera", init<std::string>() )
		ADD_PY_PROPERTY( Camera, Name )
		ADD_PY_PROPERTY_RDONLY( Camera, ID )
		ADD_PY_PROPERTY( Camera, ResolutionWidth )
		ADD_PY_PROPERTY( Camera, ResolutionHeight )
		ADD_PY_PROPERTY( Camera, FOV )
		ADD_PY_PROPERTY( Camera, NearClipPlane )
		ADD_PY_PROPERTY( Camera, FarClipPlane )
		DEF_PY_FUNCTION( Camera, setTransform )
		;
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
