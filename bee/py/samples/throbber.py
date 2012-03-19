
import sys
import os

try:
    from OpenGL.GLUT import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except:
    print 'PyOpenGL could not be imported'
    sys.exit()

class Throbber:
    def __init__(self,rate=360.0):
        self.rot = 0.0
        self.rate = rate
        self.progress = 0.0

    def update(self,time, progress):
        self.rot += time * self.rate
        self.progress = progress

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        glViewport( 0, 0, xSize, ySize )

        if self.progress > 0.0:
            # only if progress is actually progressing do we draw anything else
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()

            # to modelview & initialise
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            # draw
            glBegin( GL_QUADS )
            glColor3f(0.0, 0.2, 0.0)
            for v in [[-1,-1,0],[-1,1,0],[self.progress*2-1,1,0],[self.progress*2-1,-1,0]]:
                glVertex3d( *v )
            glEnd()

            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)


        # to projection, and initialise
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        # make it look pretty from a certain point of view
        gluPerspective(60, float(xSize) / float(ySize), 0.1, 50)

        # to modelview & initialise
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # put the camera and quad in the exact correct place, then draw
        glTranslatef(0, 0, -2)
        glRotatef(self.rot*0.2,0,0,0.3)
        glRotatef(self.rot,0,1,0)
        glBegin( GL_QUADS )
        glColor3f( 0.4, 0.9, 0.4 )
        for v in [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]:
            glVertex3d( *v )
        glEnd()

        # restore it all
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


        glFlush()
        glutSwapBuffers()

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

