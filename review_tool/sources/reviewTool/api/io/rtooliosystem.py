##
#   \namespace  reviewTool.api.io.rtooliosystem
#
#   \remarks    [desc::commented]
#   
#   \author     eric.hulser@drdstudios.com
#   \author     Dr. D Studios
#   \date       08/18/11
#

import os

from ...xml                 import XmlDocument
from ..iosystem             import IOSystem
from ..contexts.playlist    import PlaylistContext

class RToolIOSystem(IOSystem):
    def load( self, filename ):
        xdoc = XmlDocument()
        if ( not xdoc.load(filename) ):
            return None
        
        return PlaylistContext.fromXml( os.path.basename(filename), xdoc.root(), filename )
        
    def save( self, filename, clips ):
        """
                Saves the inputed clips to the given filename
                
                :param      filename:
                :type       <str>
                
                :param      clips:
                :type       <list> [ <Clip>,..]:
                
                :return     <bool>:
        """
        xdoc    = XmlDocument()
        xroot   = xdoc.setRoot( 'review' )
        xclips  = xroot.createChild('clips')
        
        for clip in clips:
            clip.toXml( xclips )
            
        return xdoc.save(filename)

IOSystem.register( 'Review Tool Files', RToolIOSystem, '.rtool', imports = True, exports = True )
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

