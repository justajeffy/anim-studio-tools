
import sys
import pimath
import math

def MakeLookAt(pos,la,up=pimath.V3f(0,1,0)):
    f = (pos - la).normalized()
    r = up.cross(f).normalized()
    u = f.cross(r).normalized() # re-cross it to make 99.999% orthogonal
    m = pimath.M44f(r.x, u.x, f.x, 0.0,
                    r.y, u.y, f.y, 0.0,
                    r.z, u.z, f.z, 0.0,
                    -r.dot(pos), -u.dot(pos), -f.dot(pos), 1.0 )
    return m

def MakeProjection(fov,aspect,zmin,zmax):
    ymax = zmin * math.tan(fov*0.5)
    xmax = ymax * aspect
    x = zmin/xmax
    y = zmin/ymax
    c = -(zmax+zmin)/(zmax-zmin)
    d = -(2.0*zmax*zmin)/(zmax-zmin)

    m = pimath.M44f(x,0,0,0,
                    0,y,0,0,
                    0,0,c,-1,
                    0,0,d,0)
    return m

def GetOrientationMatrix(m):
    x1,x2,x3 = m.value[0][0:3]
    x4,x5,x6 = m.value[1][0:3]
    x7,x8,x9 = m.value[2][0:3]
    return pimath.M33f(x1,x2,x3,x4,x5,x6,x7,x8,x9)

def ToMatrix44(m):
    x1,x2,x3 = m.value[0][0:3]
    x4,x5,x6 = m.value[1][0:3]
    x7,x8,x9 = m.value[2][0:3]
    return pimath.M44f(x1,x2,x3,0.0, x4,x5,x6,0.0, x7,x8,x9,0.0, 0.0,0.0,0.0,1.0)

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

