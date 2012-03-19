/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/glExtensions.cpp $"
 * SVN_META_ID = "$Id: glExtensions.cpp 32042 2010-05-20 00:41:59Z stephane.bertout $"
 */

#include "glExtensions.h"

#include <GL/glxew.h>
#include <GL/glew.h>

#include <iostream>

static bool s_bExtensionsInitialized = false;

bool initGLExtensions(void)
{
	if ( s_bExtensionsInitialized == false )
	{
		int err = glewInit();

		if ( GLEW_OK != err )
		{
			printf((char*)glewGetErrorString(err));
			return false;
		}

		s_bExtensionsInitialized = true;
	    return true;
    }

    return true;
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
