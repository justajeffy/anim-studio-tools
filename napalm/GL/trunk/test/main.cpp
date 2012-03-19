
#include "napalmGL/context.h"
#include "napalmGL/VectorGL.h"

#include <iostream>
#include <cassert>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

using namespace napalm_gl;



int main( 	int argc, char** argv )
{
	assert( hasOpenGL() == false );

	// minimal glut setup
	glutInit( &argc, argv );
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(150,150);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);

	assert( hasOpenGL() == true );

	// initialize gl extensions
	glewInit();

	VectorGL<float> temp;
	temp.resize(1000);

	return 0;
}
