/*
 * Copyright (c) 2009 Dr. D Studios. (Please refer to license for details)
 * SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/lib/gl/glError.cpp $"
 * SVN_META_ID = "$Id: glError.cpp 30989 2010-05-11 23:23:13Z chris.cooper $"
 */

#define GL3_PROTOTYPES
#include <GL/glew.h>
#include <GL/glut.h>
#include "../kernel/spam.h"

namespace bee
{
	int CheckGLError(char *file, int line)
	{
		GLenum glErr;
		int    retCode = 0;

		glErr = glGetError();

		while (glErr != GL_NO_ERROR)
		{
			if ( glErr = GL_INVALID_ENUM ) std::cout << "GL Error : INVALID_ENUM in File " << file << " at line: " << line << std::endl;
			else if ( glErr = GL_INVALID_VALUE ) std::cout << "GL Error : INVALID_VALUE in File " << file << " at line: " << line << std::endl;
			else if ( glErr = GL_INVALID_OPERATION ) std::cout << "GL Error : INVALID_OPERATION in File " << file << " at line: " << line << std::endl;
			//else if ( glErr = GL_STACK_OVERFLOW ) std::cout << "GL Error : STACK_OVERFLOW in File " << file << " at line: " << line << std::endl;
			//else if ( glErr = GL_STACK_UNDERFLOW ) std::cout << "GL Error : STACK_UNDERFLOW in File " << file << " at line: " << line << std::endl;
			else if ( glErr = GL_OUT_OF_MEMORY ) std::cout << "GL Error : OUT_OF_MEMORY in File " << file << " at line: " << line << std::endl;
			else std::cout << "GL Error : Unknown" << file << " at line: " << line << std::endl;

			retCode = 1;
			glErr = glGetError();
		}

		return retCode;
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
