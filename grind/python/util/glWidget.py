
import sys
import math
from pimath import *
from PyQt4 import QtCore, QtGui, QtOpenGL
from camera import Camera

import grind

#-----------------------------------------------------------------------------
from rodin import logging
log = logging.get_logger('grind.mangle.gl_widget')

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    app = QtGui.QApplication(sys.argv)
    QtGui.QMessageBox.critical(None, "mangle", "PyOpenGL must be installed to run this example.")
    sys.exit(1)


class GLWidget(QtOpenGL.QGLWidget):
    xRotationChanged = QtCore.pyqtSignal(int)
    yRotationChanged = QtCore.pyqtSignal(int)
    zRotationChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.renderable = None
        self.object = 0
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.width = 0
        self.height = 0
        self.display_selection_marker = False
        self.selection_marker_bbox = grind.BBox()
        self.selection_marker_bbox.set_colour(0xFF0000) # R G B

        self.lastPos = QtCore.QPoint()

        self.backgroundColour = QtGui.QColor.fromCmykF(0.28, 0.28, 0.28, 0.0)
        self.foregroundColour = QtGui.QColor.fromCmykF(0.7, 0.7, 0.7, 0.0)

        self.dist = 1.0
        self.up = 1.0
        self.drawGrid = True
        self.drawDefaultObject = True
        self.followBBox = False
        self.moveGrid = False
        self.camera = Camera()
        self.frameView()

    def setFollowBBox(self,follow):
        self.followBBox = follow
        self.updateGL()

    def setCenterBBox(self,centered):
        self.moveGrid = not centered
        self.updateGL()

    def setRenderable(self,renderable,callframeview=True):
        self.renderable = renderable
        if callframeview == True:
            self.frameView()
        self.resizeGL(self.width,self.height)
        self.updateGL()

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(640, 480)

    def frameView(self,update=False):
        if self.renderable is None:
            self.camera.frame(V3f(0,0,0),1)
            if update:
                self.updateGL()
            return
        bb = self.renderable.getBounds()
        height = bb.size().y
        c = bb.center()
        center = V3f(c.x,c.y,c.z)
        self.camera.frame(center,height)

        self.up = height*1.2
        self.dist = self.camera.distanceNeeded(height)
        if update:
            self.updateGL()

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.xRotationChanged.emit(angle)
            self.updateGL()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.yRotationChanged.emit(angle)
            self.updateGL()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.zRotationChanged.emit(angle)
            self.updateGL()

    def initializeGL(self):
        self.qglClearColor(self.foregroundColour.dark())
        self.object = self.makeObject()
        self.grid = self.makeGrid()
        glShadeModel(GL_FLAT)
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef( 0, -self.up, -self.dist )
        if self.followBBox:
            self.frameView(False)
        self.camera.update(1/60.0)
        if self.drawGrid:
            move = self.moveGrid and self.renderable is not None
            if move:
                glPushMatrix()
                center = self.renderable.getBounds().center()
                glTranslatef(round(center.x/5)*5,round(center.y/5)*5,round(center.z/5)*5)
            glCallList(self.grid)
            if move:
                glPopMatrix()

        if self.renderable is None:
            if self.drawDefaultObject:
                glCallList(self.object)
        else:
            self.renderable.update()
            self.renderable.render()

        if self.display_selection_marker == True:
            x = self.lastPos.x()
            y = self.height - self.lastPos.y()
            z = (GLfloat * 1)(0)
            glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, z)

            if z[0] < 1: # ignore void click
                proj = (ctypes.c_double*16)()
                proj = glGetDoublev(GL_PROJECTION_MATRIX)
                model = (ctypes.c_double*16)()
                model = glGetDoublev(GL_MODELVIEW_MATRIX)
                (wx,wy,wz) = gluUnProject( x,y,z[0], model, proj, (0, 0, self.width, self.height) ) # model proj view

                scale = (self.camera.pos - V3f(wx,wy,wz)).length() * 0.0025
                self.selection_marker_bbox.min = V3f(wx - scale, wy - scale, wz - scale)
                self.selection_marker_bbox.max = V3f(wx + scale, wy + scale, wz + scale)
                glDisable(GL_DEPTH_TEST)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                self.selection_marker_bbox.render(1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glEnable(GL_DEPTH_TEST)


    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        side = min(width, height)
        if side < 0:
            return
        self.camera.aspect = float(self.width)/float(self.height)

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(35, float(self.width)/float(self.height), 0.01, 100)
        glMatrixMode(GL_MODELVIEW)


    def target_selection(self):
        if self.display_selection_marker == True:
            self.camera.lookat = V3f(self.selection_marker_bbox.center().x, self.selection_marker_bbox.center().y, self.selection_marker_bbox.center().z)
            newdir = (self.camera.pos - self.camera.lookat).normalized()
            self.camera.pos = self.camera.lookat + newdir * self.camera.dist
            self.display_selection_marker = False


    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        self.camera.mouseButton(event.button(), True, self.lastPos.x(), self.lastPos.y())
        self.updateGL()


    def mouseReleaseEvent(self, event):
        self.camera.mouseButton(event.button(), False, self.lastPos.x(), self.lastPos.y())
        self.updateGL()


    def mouseMoveEvent(self, event):
        self.camera.mouseMotion(event.x(), event.y())
        self.updateGL()
        self.lastPos = event.pos()

    def wheelEvent(self,event):
        self.updateGL()


    def makeObject(self):
        genList = glGenLists(1)
        glNewList(genList, GL_COMPILE)

        NumSectors = 13
        Length = 10.0
        LengthSec = 25

        Outer = 0.5
        Inner = 0.4
        ZInner = -Length/2.0
        ZOuter = ZInner+0.04
        ZInner = 0.01
        ZOuter = -0.01

        for j in range(LengthSec+1):

            glBegin(GL_QUADS)
            for i in range(NumSectors):
                angle1 = (i * 2 * math.pi) / NumSectors
                x5 = Outer * math.sin(angle1)
                y5 = Outer * math.cos(angle1)
                x6 = Inner * math.sin(angle1)
                y6 = Inner * math.cos(angle1)

                angle2 = ((i + 1) * 2 * math.pi) / NumSectors
                x7 = Inner * math.sin(angle2)
                y7 = Inner * math.cos(angle2)
                x8 = Outer * math.sin(angle2)
                y8 = Outer * math.cos(angle2)

                #self.quad(x5, y5, x6, y6, x7, y7, x8, y8,  ZOuter,  ZInner)
                self.extrude(x6, y6, x7, y7, ZOuter, ZInner)
                #self.extrude(x8, y8, x5, y5, ZOuter, ZInner)

            glEnd()
            #glTranslate(0,0,Length/LengthSec)
            glRotate(6.8,0,1.91231233,0)

        glEndList()

        return genList

    def quad(self, x1, y1, x2, y2, x3, y3, x4, y4, z1, z2):
        self.qglColor(self.backgroundColour)

        glVertex3d(x1, y1, z2)
        glVertex3d(x2, y2, z2)
        glVertex3d(x3, y3, z2)
        glVertex3d(x4, y4, z2)

        glVertex3d(x4, y4, z1)
        glVertex3d(x3, y3, z1)
        glVertex3d(x2, y2, z1)
        glVertex3d(x1, y1, z1)

    def extrude(self, x1, y1, x2, y2, z1, z2):
        self.qglColor(self.backgroundColour.dark(250 + int(100 * x1)))

        glVertex3d(x1, y1, z1)
        glVertex3d(x2, y2, z1)
        glVertex3d(x2, y2, z2)
        glVertex3d(x1, y1, z2)

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def makeGrid(self):
        genList = glGenLists(1)
        glNewList(genList, GL_COMPILE)
        glBegin(GL_LINES)

        self.qglColor(self.backgroundColour.dark(150))

        self.qglColor(QtGui.QColor(70,70,80))
        size = 10.0
        count = 10.0
        xs = []
        ys = []
        for x in range(int(count)):
            xpos = (x/count-0.5)*size
            xs.append(xpos)
            for y in range(int(count)):
                ypos = (y/count-0.5)*size
                ys.append(ypos)
                a = ( xpos,0, ypos)
                b = ( xpos,0,-ypos)
                c = (-xpos,0,-ypos)
                d = (-xpos,0, ypos)
                glVertex3d(*a)
                glVertex3d(*b)
                glVertex3d(*d)
                glVertex3d(*c)
                glVertex3d(*a)
                glVertex3d(*d)
                glVertex3d(*b)
                glVertex3d(*c)

        self.qglColor(QtGui.QColor(54,54,54))
        size = 10.0
        count = 100.0
        for x in range(int(count)):
            xpos = (x/count-0.5)*size
            if xpos in xs: continue
            for y in range(int(count)):
                ypos = (y/count-0.5)*size
                if ypos in ys: continue
                a = ( xpos,0, ypos)
                b = ( xpos,0,-ypos)
                c = (-xpos,0,-ypos)
                d = (-xpos,0, ypos)
                glVertex3d(*a)
                glVertex3d(*b)
                glVertex3d(*d)
                glVertex3d(*c)
                glVertex3d(*a)
                glVertex3d(*d)
                glVertex3d(*b)
                glVertex3d(*c)

        glEnd()
        glEndList()

        return genList


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

