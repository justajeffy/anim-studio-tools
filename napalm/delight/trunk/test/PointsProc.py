import pimath
import napalm.core as n
import napalmDelight as nd
import ribclient as ri
import random


class PointsProc():

    def __init__(self, centre, dim, point_radius):
        self.centre = centre
        self.dim = dim
        self.point_radius = point_radius

    def bound(self):
        return [ \
            self.centre[0] - self.dim - self.point_radius, self.centre[0] + self.dim + self.point_radius, \
            self.centre[1] - self.dim - self.point_radius, self.centre[1] + self.dim + self.point_radius, \
            self.centre[2] - self.tim - self.point_radius, self.centre[2] + self.dim + self.point_radius ]

    def render(self, detailSize):

        # set up some buffers
        count = 20000

        P = []
        Cs = []
        w = []

        random.seed(997)
        for i in range(count):
            P.append(pimath.V3f(random.uniform(-self.dim, self.dim), random.uniform(-self.dim, self.dim), random.uniform(-self.dim, self.dim)) )
            Cs.append(pimath.V3f(random.uniform(0.05, 1), random.uniform(0.05, 1), random.uniform(0.05, 1)))
            w.append(random.uniform(0.5 * self.point_radius, self.point_radius))

        # form a table from our buffers with required renderman info
        t = n.ObjectTable()
        t['P'] = n.V3fBuffer(count, pimath.V3f(0, 0, 0))
        t['P'].contents = P
        t['P'].attribs['token'] = 'P'

        if 1:
            # colour per point
            t['Cs'] = n.V3fBuffer(count, pimath.V3f(0, 0, 0))
            t['Cs'].contents = Cs
            t['Cs'].attribs['token'] = 'vertex color Cs'
        else:
            # constant colour across all points
            t['Cs'] = n.AttribTable()
            t['Cs']['value'] = pimath.V3f(1,0,0)
            t['Cs']['token'] = 'constant color Cs'

        if 0:
            # varying width
            t['width'] = n.FloatBuffer(count, 0.0)
            t['width'].contents = w
            t['width'].attribs['token'] = 'varying float width'
        else:
            if 1:
                # constants either as attrib table
                t['width'] = n.AttribTable()
                t['width']['value'] = 0.05
                t['width']['token'] = 'constant float constantwidth'
            else:
                # or buffer of length 1
                t['width'] = n.FloatBuffer(1, 0.03)
                t['width'].attribs['token'] = 'constant float constantwidth'

        # render
        ri.TransformBegin()
        ri.Translate(self.centre[0], self.centre[1], self.centre[2])
        nd.points(t)
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

