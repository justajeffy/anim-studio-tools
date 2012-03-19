


#include "napalmGL/context.h"

#include <GL/glx.h>
#include <GL/glu.h>

#include <iostream>

bool napalm_gl::hasOpenGL()
{
	return glXGetCurrentContext() != NULL;
}



bool napalm_gl::noErrorsGL()
{
	GLenum errCode;

	if ((errCode = glGetError()) != GL_NO_ERROR) {
		std::cerr << "OpenGL Error at " << __FILE__ << ":" << __LINE__ << ": " << gluErrorString(errCode) << std::endl;
		return false;
	}

	return true;
}

