#!/usr/bin/env python2.5

SVN_META_HEADURL = "$HeadURL: http://svn/drd/apps/bee/trunk/py/samples/flipbook.py $"
SVN_META_ID = "$Id: flipbook.py 17926 2009-11-30 06:47:09Z david.morris $"

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
from throbber import Throbber
from testScene import toRender
from PIL import Image, ImageQt, ImageOps

rt = ''
throbber = ''
scene = ''
frameNumber = 0
renderSize = (1920,1080)

def display():
    global scene
    global rt
    global frameNumber
    global throbber


    xSize,ySize = renderSize
    rt.use()
    shouldContinue, progress = scene.update(1/60.0)
    scene.render()
    glFlush()
    arr = glReadPixels(0,0,xSize,ySize,GL_RGBA,GL_BYTE,None)
    arr1 = arr.ravel()
    img = Image.frombuffer('RGBA', (xSize,ySize), arr1, 'raw', 'RGBA', 0, 1)
    img.save('out/frame_%06i.jpg'%frameNumber, "JPEG")
    frameNumber+=1

    rt.release()
    # print 'Progress: %3.1f%%' % (progress*100.0)
    if not shouldContinue:
        sys.exit(0)

    throbber.update(1/60.0, progress)
    throbber.draw()


def init():
    global scene
    global rt
    global throbber
    lA = 0.8
    lD = 1
    lS = 1

    glClearColor (0.3, 0.3, 0.3, 0.3)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1)
    glLightfv(GL_LIGHT0, GL_POSITION, [4, 4, 4, 1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [lA, lA, lA, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [lD, lD, lD, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [lS, lS, lS, 1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.2, 0.2, 0.7, 1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)
    rt = BEE.RenderTarget()
    rt.create(*renderSize)
    throbber = Throbber()
    scene = toRender()
    scene.init(*renderSize)

print 'Using BEE: %s / %s' % (BEE.getVersion(), SVN_META_ID)
glutInit(sys.argv)
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
glutInitWindowSize(150,150)
#glutInitWindowPosition(100, 100)
glutCreateWindow(sys.argv[0])
init()
glutDisplayFunc(display)
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

