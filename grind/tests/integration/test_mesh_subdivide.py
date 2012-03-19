#! /usr/bin/env python2.5

import sys, time
from mouseInteractor import MouseInteractor

try:
	from OpenGL.GLUT import *
	from OpenGL.GL import *
	from OpenGL.GLU import *
except:
	print ''' Error: PyOpenGL nicht intalliert !!'''
	sys.exit()

import grind
mesh = None
mesh_subd = None
subdivider = None
prev_t = 0
mesh_shader = None
mesh_tex = None

def display():
	"""Glut display function."""
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)

	if 0:
		gluPerspective(60, float(xSize) / float(ySize), 0.1, 5000)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glTranslatef(-184,-40,-293)
	else:
		gluPerspective(35, float(xSize) / float(ySize), 0.2, 100)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glTranslatef( 0, -5, -20 )

	#glTranslatef(-180,-45,-293)
	global mouseInteractor
	mouseInteractor.applyTransformation()

	global subdivider
	global mesh
	global mesh_shader
	global mesh_tex
	global mesh_subd

	mesh_shader.use()
	mesh_tex.use(0, mesh_shader, 0, -1)
	#subdivider.update( mesh )
	mesh_subd.render(1)
	#mesh.render(1)
	mesh_tex.un_use(0)
	mesh_shader.un_use()

	glutSwapBuffers()
	this_t = time.time()
	global prev_t
	fps = 1.0 / (this_t-prev_t)
	prev_t = this_t
	glutSetWindowTitle( 'fps: %.2f' % fps );

def init():
	"""Glut init function."""
	glClearColor (0, 0, 0, 0)
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_DEPTH_TEST)
	glShadeModel(GL_SMOOTH)
	glEnable(GL_CULL_FACE);
#	glEnable( GL_LIGHTING )
#	glEnable( GL_LIGHT0 )
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0)
	glLightfv(GL_LIGHT0, GL_POSITION, [0, 200, 400, 1])
	lA = 0.8
	glLightfv(GL_LIGHT0, GL_AMBIENT, [lA, lA, lA, 1])
	lD = 1
	glLightfv(GL_LIGHT0, GL_DIFFUSE, [lD, lD, lD, 1])
	lS = 1
	glLightfv(GL_LIGHT0, GL_SPECULAR, [lS, lS, lS, 1])
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.0, 0.0, 0.2, 1])
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.0, 0.0, 0.7, 1])
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)
	global mouseInteractor
	mouseInteractor = MouseInteractor(.01, 1)

	global mesh
	global mesh_subd
	mesh = grind.DeviceMesh()
	mesh_subd = grind.DeviceMesh()
	#mesh.read( 'obj/rest.obj')
	#mesh.read( 'obj/plane.obj')
	#mesh.read( 'obj/open_box.obj')
	#mesh.read( 'obj/plane_with_triangle.obj')
	mesh.read( 'obj/lep_seal_adult.obj')
	#mesh.read( 'obj/lep_seal_adult_tri_tail.obj')
	#mesh.read('obj/single_quad.obj')

	global mesh_shader
	mesh_shader = grind.Program()
	mesh_shader.read('glsl/blinn.vs.glsl', 'glsl/blinn.fs.glsl')
	#mesh_shader.read('glsl/test_150.vs.glsl', 'glsl/test_150.fs.glsl')

	global mesh_tex
	mesh_tex = grind.Texture()
	#mesh_tex.read('maps/glr_todl_fur_body_bcolor_v14.tif')
	mesh_tex.read('maps/white.tif')

	global subdivider
	subdivider = grind.MeshSubdivide()
	subdivider.set_iterations(3)
	subdivider.process( mesh, mesh_subd )
	grind.info()

def keyboard( key, a, b ):
	# exiting is painfully slow if memory isn't de-allocated correctly
	if (key == 27) or (key == 'q'):
		sys.exit()

# we should be able to initialize gl context after loading grind
# due to lazy singleton initialization of gl extensions etc
glutInit(sys.argv)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(500, 500)
glutInitWindowPosition(100, 100)
glutCreateWindow(sys.argv[0])

init()
mouseInteractor.registerCallbacks()
glutKeyboardFunc(keyboard);
glutIdleFunc( display )
glutMainLoop()
















# Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)
#
# This file is part of anim-studio-tools.
#
# anim-studio-tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# anim-studio-tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.

