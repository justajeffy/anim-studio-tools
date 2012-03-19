# helper class for interactive object motion 
# 
# Copyright (C) 2007  "Peter Roesch" <Peter.Roesch@fh-augsburg.de>
#
# This code is licensed under the PyOpenGL License.
# Details are given in the file license.txt included in this distribution.

import sys

try:
  from OpenGL.GLUT import *
  from OpenGL.GL import *
  from OpenGL.GLU import *
except:
  print ''' Error: PyOpenGL not installed properly !!'''
  sys.exit(  )

class InteractionMatrix ( object ):
	"""Class holding a matrix representing a rigid transformation.

	The current OpenGL is read into an internal variable and
	updated using rotations and translations given by
	user interaction."""

	def __init__( self ):
		self.__currentMatrix = None
		self.reset( )

	def reset( self ):
		"""Initialise internal matrix with identity"""
		glPushMatrix( )
		glLoadIdentity( )
		self.__currentMatrix = glGetFloatv( GL_MODELVIEW_MATRIX )
		glPopMatrix( )

	def addTranslation( self, tx, ty, tz ):
		"""Concatenate the internal matrix with a translation matrix"""
		glPushMatrix( )
		glLoadIdentity( )
		glTranslatef(tx, ty, tz)
		glMultMatrixf( self.__currentMatrix )
		self.__currentMatrix = glGetFloatv( GL_MODELVIEW_MATRIX )
		glPopMatrix( )

	def addRotation( self, ang, rx, ry, rz ):
		"""Concatenate the internal matrix with a translation matrix"""
		glPushMatrix( )
		glLoadIdentity( )
		glRotatef(ang, rx, ry, rz)
		glMultMatrixf( self.__currentMatrix )
		self.__currentMatrix = glGetFloatv( GL_MODELVIEW_MATRIX )
		glPopMatrix( )

	def getCurrentMatrix( self ):
		return self.__currentMatrix
	
if __name__ == '__main__' :
	glutInit( sys.argv )
	glutCreateWindow( sys.argv[0] )
	m=InteractionMatrix()
	print m.getCurrentMatrix( )
	m.addTranslation(1,2,3)
	print m.getCurrentMatrix( )
	m.addRotation(30,0,0,1)
	print m.getCurrentMatrix( )
	m.addTranslation(1,2,3)
	print m.getCurrentMatrix( )

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

