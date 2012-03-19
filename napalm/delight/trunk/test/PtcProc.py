import pimath
import napalm.core as n
import napalmDelight as nd
import ribclient as ri
import random


class PtcProc():

    def __init__(self, path):
        self.path = path
        print 'PtcProc.__init__()'

    def bound(self):
        return [ -1000, 1000, -1000, 1000, -1000, 1000 ]

    def render(self, detailSize):
        print 'PtcProc.render()'
        ptc = nd.load( self.path )

        #ptc['firstN'] = 20000

        # render
        nd.points(ptc)

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

