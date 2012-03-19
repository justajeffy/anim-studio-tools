
import sys
import os

try:
    from OpenGL.GLUT import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except:
    print 'PyOpenGL could not be imported'
    sys.exit()

import BEE

class toRender:
    def __init__(self):
        self.program = ''
        self.tex = ''
        self.rot = 0.0
        self.rotInc = 1.0
        self.xSize = 100.0
        self.ySize = 100.0

    def init(self, xSize, ySize):
        self.program = BEE.Program()
        self.program.read( 'glsl/blinn.vs.glsl', 'glsl/blinn.fs.glsl')
        self.tex = BEE.Texture()
        self.tex.read( 'data/happy_feet.jpg' )
        self.xSize = float(xSize)
        self.ySize = float(ySize)

    def update(self, elapsed):
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.xSize / self.ySize, 0.1, 50)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, -0.0, -3)
        glRotatef(self.rot, 0, 1, 0)
        self.rot += self.rotInc
        return self.rot < 360.0, self.rot/360.0

    def render(self):
        self.program.use()
        self.tex.use(0, self.program, 0, -1)
        steps = 36
        scale = 1
        xlate = 0.1
        for i in range( steps ):
            glPushMatrix()
            glScalef(scale,scale,scale)
            glRotatef((360.0*i)/steps, 0.0, 0.8, 0.2)
            glTranslatef(xlate, 0, xlate*0.4)
            glutSolidTeapot( 1.0 )
#            glBegin( GL_QUADS )
#            for v in [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]:
#                glTexCoord2f( *v[:2] )
#                glVertex3d( *v )
#            glEnd()
            glPopMatrix()
        self.tex.release(0)
        self.program.release()



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

