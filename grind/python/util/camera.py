
import sys
from pimath import *
import math
from . import util # stoopid python thinks it's the standard util one
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt4 import QtCore


def _Tumble(cam,delta):
    if delta.length() < 0.001:
        return False
    euler = V3f(-delta.y,delta.x,0.0)
    rot = util.GetOrientationMatrix(util.MakeLookAt(cam.lookat,cam.pos,cam.up))
    eulerMat = M44f()
    eulerMat.setEulerAngles(euler)
    eulerMat = util.GetOrientationMatrix(eulerMat)
    xform = rot*(eulerMat.transposed()*rot.transposed())
    pos = cam.lookat + (cam.pos-cam.lookat) * xform
    dir = (cam.lookat-pos).normalized()
    if abs(dir.y) < 0.98:
        cam.pos = pos
    cam.dist = (cam.pos-cam.lookat).length()
    return True

def _Move(cam,delta,staticLA=False):
    if delta.length() < 0.001:
        return False
    delta = V3f(-delta.x,delta.y,delta.z)*(cam.dist*0.4)
    mtx = util.MakeLookAt(cam.pos,cam.lookat,cam.up)
    mtxt = util.GetOrientationMatrix(mtx).transposed()
    offset = V3f(*mtxt.value[0])*delta.x + V3f(*mtxt.value[1])*delta.y + V3f(*mtxt.value[2])*delta.z

    if delta.z == 0:
        cam.pos = cam.pos+offset
        cam.lookat = cam.lookat+offset
    else:
        zD = (1.0+delta.z*3.0)
        zD = zD if zD > 0.0 else 1.0
        v = cam.pos-cam.lookat
        if v.length() < 0.1:
            v = v.normalized() * 0.1
        elif v.length() < 1.5:
             v *= zD
        cam.pos = cam.pos+offset
        #cam.lookat = cam.pos-v
    cam.dist = (cam.pos-cam.lookat).length()

class Camera:
    def __init__(self):
        self.updateViewProj = True
        self.view = M44f()
        self.proj = M44f()
        self.pos = V3f(0,0,1)
        self.up = V3f(0,1,0)
        self.lookat = V3f(0,0,0)
        self.zmin = 0.1
        self.zmax = 100.0
        self.tumbleSpeed = 0.1 / 60.0
        self.mouseMoveSpeed = 0.3 / 60.0
        self.mouseScrollMoveSpeed = 1.5 / 60.0
        self.mouseButtonPressed = None
        self.oldMousePos = V3f(0,0,0)
        self.dist = (self.pos-self.lookat).length()
        self.aspect = 1.33
        self.haperture = 24.892
        self.vaperture = self.haperture / self.aspect
        self.setMM(35.0)

    def setMM(self,mm):
        self.fov = math.atan2((self.haperture/2.0),mm)*2.0
        self.fovY = math.atan2((self.vaperture/2.0),mm)*2.0

    def getMM(self):
        return (self.haperture/2.0) / math.tan(self.fov/2.0)

    def distanceNeeded(self,size):
        h = (size/2.0) / math.tan(self.fov/2.0)
        v = (size/2.0) / math.tan(self.fovY/2.0)
        return max(h,v)

    def mouseButton(self, button, pressed, x, y):
        self.mouseButtonPressed = ( button if pressed else None )
        self.oldMousePos = V3f(x,y,0)

    def mouseMotion(self, x, y):
        delta = V3f(x,y,0)-self.oldMousePos
        tumbleAmount = V3f(0,0,0)
        moveAmount = V3f(0,0,0)
        if self.mouseButtonPressed == QtCore.Qt.LeftButton:
            tumbleAmount += delta*self.tumbleSpeed
        elif self.mouseButtonPressed == QtCore.Qt.MidButton:
            moveAmount.x = delta.x*self.mouseMoveSpeed
            moveAmount.y = delta.y*self.mouseMoveSpeed
        elif self.mouseButtonPressed == QtCore.Qt.RightButton:
            moveAmount.z -= delta.x*self.mouseMoveSpeed
            moveAmount.z += delta.y*self.mouseMoveSpeed

        _Tumble(self,tumbleAmount)
        _Move(self,moveAmount)
        self.oldMousePos = V3f(x,y,0)

    def update(self,elapsed=1.0/60.0):
        if self.updateViewProj:
            self.view = util.MakeLookAt(self.pos,self.lookat,self.up)
            self.proj = util.MakeProjection(self.fov,self.aspect,self.zmin,self.zmax)
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf( self.proj.value );
        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf( self.view.value );
        #dist = (self.pos-self.lookat).length()
        #print "p: %5.2f, %5.2f, %5.2f ; la: %5.2f, %5.2f, %5.2f ; u: %5.2f, %5.2f, %5.2f ; D: %5.2f" % (self.pos.x,
        #    self.pos.y, self.pos.z, self.lookat.x, self.lookat.y, self.lookat.z, self.up.x, self.up.y, self.up.z, dist)

    def frame(self,center,height):
        dir = (self.pos-self.lookat).normalized()
        dist = (self.pos-self.lookat).length()
        self.lookat = center
        self.pos = center + dir*self.distanceNeeded(height)*1.4




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

