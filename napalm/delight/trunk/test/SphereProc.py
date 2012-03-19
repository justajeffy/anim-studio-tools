import pimath
import napalm.core as n
import napalmDelight as nd
import ribclient as ri
import random


class SphereProc():

    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius

    def bound(self):
        return [ \
            self.centre[0] - self.dim - self.point_radius, self.centre[0] + self.dim + self.point_radius, \
            self.centre[1] - self.dim - self.point_radius, self.centre[1] + self.dim + self.point_radius, \
            self.centre[2] - self.tim - self.point_radius, self.centre[2] + self.dim + self.point_radius ]

    def render(self, detailSize):

        # form a table from our buffers with required renderman info
        t = n.ObjectTable()
        t['radius'] = self.radius
        t['zmin'] = -self.radius
        t['zmax'] = self.radius
        t['thetamax'] = 360.0

        # constant colour
        t['Cs'] = n.AttribTable()
        t['Cs']['value'] = pimath.V3f(1,0.5,0)
        t['Cs']['token'] = 'constant color Cs'

        # render
        ri.TransformBegin()
        ri.Translate(self.centre[0], self.centre[1], self.centre[2])
        nd.sphere(t)
        ri.TransformEnd()

    def free(self):
        pass


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

