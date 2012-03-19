##
#   \namespace  reviewTool.api.io.rviosystem
#
#   \remarks    [desc::commented]
#   
#   \author     eric.hulser@drdstudios.com
#   \author     Dr. D Studios
#   \date       08/18/11
#

import os.path
import rv_tools

from ..iosystem             import IOSystem
from ..clip                 import Clip
from ..version              import Version
from ..contexts.playlist    import PlaylistContext

#--------------------------------------------------------------------------------

class RVIOSystem(IOSystem):
    def __init__( self, systemName, fileType = '', imports = False, exports = False ):
        super(RVIOSystem,self).__init__( systemName, fileType, imports, exports )
        
        self._version   = '1.0'
        self._fps       = 24
    
    def load( self, filename ):
        # load the file from rv_tools
        rv_file     = rv_tools.open_file( filename, 'r' )
        clips       = rv_file.get_sequence().list_clips()
        
        # patch data for xml files
        lookupBy = 'sg_tank_address'
        if ( filename.endswith( '.xml' ) ):
            rv_tools.util.patch_fcp_data(clips)
            lookupBy = 'code'
        
        sg_versions         = []
        
        # load the various clips into the context
        for i, clip in enumerate(clips):
            code = Version.findCode( clip.left_source_path )
            
            try:
                addr = clip.get_meta_data('Tank Revision')['value']
            except:
                addr = None
            
            # add the look up version
            sg_versions.append({'type':'Version','code':code,'sg_tank_address':addr, 'playbackStart': clip.frame_in, 'playbackEnd': clip.frame_out, 'audioOverridden': clip.use_clip_audio})
        
        name = os.path.basename(filename)
        data = {'filename': filename, 'versions': sg_versions, 'lookupBy': lookupBy }
        
        return PlaylistContext( name, data = data )
    
    def save( self, filename, clips ):
        # create the rv sequence for the inputed clips
        rv_sequence = Clip.generateRVSequence( clips )
        
        # make sure we have properly converted clips
        if ( not rv_sequence.list_clips() ):
            return False
        
        # save the file out
        rv_tools.save_file( rv_sequence, filename )
        return True

# register the system
IOSystem.register( 'RV Files', RVIOSystem, '.rv', imports = True, exports = True )
IOSystem.register( 'Final Cut XML', RVIOSystem, '.xml', imports = True )
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

