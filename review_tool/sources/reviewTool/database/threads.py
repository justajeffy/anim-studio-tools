##
#   \namespace  reviewTool.database.threads
#
#   \remarks    Defines specific database lookup threads
#               that can be used to lookup information in
#               a multi-threaded way
#   
#   \author     Dr. D Studios
#   \date       07/28/11
#

from PyQt4.QtCore   import QThread

from .database      import Database

class EntityVersionsThread(QThread):
    def __init__( self, project, fields, filter ):
        super(QThread,self).__init__()
        
        self._filter     = filter
        self._project    = project
        self._fields     = fields
        self._versions   = []
        
    def run( self ):
        # create a connection to a new database
        db = Database()
        
        # create filters
        filters = []
        
        # support older lookups, before 'SceneShot' was defined as 'Shot'
        filter = str(self._filter).replace( 'SceneShot', 'Shot' )
        
        filters.append(['sg_tank_address','contains',filter])
        filters.append(['sg_tank_address','contains','ShotRender'])
        filters.append(['sg_status_list','is_not','omt'])
        filters.append(['project','is',self._project])
        
        self._versions = db.session().find('Version',filters,self._fields)
    
    def versions( self ):
        return self._versions
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

