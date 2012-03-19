##
#   \namespace  reviewTool.contexts.sequence
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       07/27/11
#

import re

from ...database    import db
from ...            import settings

from ..context      import Context
from ..entity       import Entity
from ..version      import Version

class SequenceContext(Context):
    def __init__( self, name, data = {} ):
        super(SequenceContext,self).__init__(name)
        
        # set the custom properties
        self._shotgunId = data['id']
        self._sortOrder  = data.get('sg_sort_order')
        
        # make sure there is a cut order for this entity
        if ( self._sortOrder == None ):
            self._sortOrder = SequenceContext.generateSortOrder(name)
        
    def collectEntities( self ):
        # collect all the shots that are part of this scene from shotgun
        sg_shots    = db.findShotsByScene(self.name())
        shot_status = settings.enabledShots()
        
        entities = []
        for sg_shot in sg_shots:
            if ( sg_shot['sg_status_list'] and not sg_shot['sg_status_list'] in shot_status ):
                continue
                
            # create a new entity based on the given shot record
            entity = Entity( self, 'SceneShot', sg_shot['code'], sg_shot['id'], sg_shot['sg_status_list'] )
            
            # load the cut order
            sortOrder    = sg_shot.get('sg_cut_order')
            if ( sortOrder == None ):
                sortOrder = SequenceContext.generateSortOrder(entity.name())
                
            entity.setSortOrder( sortOrder )
            
            # add the shot entity to the output list
            entities.append( entity )
            
        entities.sort( lambda x,y: cmp( x._sortOrder, y._sortOrder ) )
        return entities
    
    def shotgunId( self ):
        return self._shotgunId
    
    @staticmethod
    def contexts():
        sg_scenes   = db.findScenes()
        sequences   = [ SequenceContext(sg_scene['code'],sg_scene) for sg_scene in sg_scenes ]
        sequences.sort( lambda x,y: cmp( x._sortOrder,y._sortOrder ) )
        return sequences
    
    @staticmethod
    def generateSortOrder( name ):
        number   = 0
        results     = re.match( '(\d+)', name )
        if ( results ):
            number   = int(results.groups()[0])
        return (10000000 + number)
        
#--------------------------------------------------------------------------------

Context.registerType( 'Sequence', SequenceContext )
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

