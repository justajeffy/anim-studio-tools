##
#   \namespace  reviewTool.api.io.csviosystem.py
#
#   \remarks    [desc::commented]
#   
#   \author     eric.hulser@drdstudios.com
#   \author     Dr. D Studios
#   \date       08/18/11
#

import datetime
import os.path
import rv_tools
import tank

from ..iosystem             import IOSystem
from ..clip                 import Clip
from ..version              import Version
from ..contexts.playlist    import PlaylistContext
from ...kernel              import core
from ...database            import db

#--------------------------------------------------------------------------------

class CSVIOSystem(IOSystem):
    def __init__( self, systemName, fileType = '', imports = False, exports = False ):
        super(CSVIOSystem,self).__init__( systemName, fileType, imports, exports )
        
        self._version   = '1.0'
        self._fps       = 24
    
    def load( self, filename ):
        # load the file from rv_tools
        csv_file    = open(filename, 'r')
        lines       = csv_file.readlines()
        csv_file.close()
        
        if  ( not lines ):
            core.warn('No data was found for: %s' % filename)
            return None
        
        header = lines[0].split(',')
        
        if ( not 'Shot' in header ):
            core.warn('Could not find the "Shot" column for: %s' % filename)
            return None
        
        index = header.index('Shot')
        sg    = db.session()
        
        dept_order = ['comp', 'light', 'anim']
        
        filters = []
        filters.append(['project', 'is', { 'id': 2, 'type': 'Project'}])
        filters.append(['created_at', 'greater_than', datetime.datetime.now() - datetime.timedelta(30)])
        filters.append(['sg_tank_address', 'contains', 'ReviewType(creative)'])
        
        fields  = Version.ShotgunFields
        
        order   = [ {'field_name': 'created_at', 'direction': 'asc'} ]
        
        sg_versions = []
        for line in lines[1:]:
            shot_id = line.split(',')[index]
            for dept in dept_order:
                shot_filters = [['code', 'contains', shot_id]]
                shot_filters.append( ['code', 'contains', dept] )
                shot_filters += filters
                
                sg_version = sg.find_one('Version', shot_filters, fields, order)
                if ( sg_version ):
                    sg_versions.append(sg_version)
                    break
        
        # patch data for xml files
        lookupBy = 'sg_tank_address'
        name = os.path.basename(filename)
        data = {'filename': filename, 'versions': sg_versions, 'lookupBy': lookupBy }
        
        return PlaylistContext( name, data = data )
    
# register the system
IOSystem.register( 'CSV Files', CSVIOSystem, '.csv', imports = True )
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

