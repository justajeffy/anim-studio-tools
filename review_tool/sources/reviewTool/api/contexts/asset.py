##
#   \namespace  reviewTool.api.contexts.asset
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/02/11

from ..context      import Context
from ..entity       import Entity
from ...database    import db

class AssetContext(Context):
    ShotgunAssetTypeMap = {
        'Character':    [ 'Character', 'Surf Var' ],
        'Prop':         [ 'Prop' ],
        'Stage':        [ 'Stage' ],
        'Environment':  [ 'Environment' ],
        'Skydome':      [ 'Lighting' ]
    }
    
    def collectEntities( self ):
        # initialize query data
        
        filters     = []
        filters.append( ['sg_status_list','is_not','omt'] )
        filters.append( ['project','is',db.project()] )
        
        fields      = ['code','assets','sg_asset_type','sg_status_list']
        
        sg_assets   = []
        for sg_type in AssetContext.ShotgunAssetTypeMap[self.name()]:
            sg_filters = [['sg_asset_type','is',sg_type]]
            if ( sg_type == 'Lighting' ):
                sg_filters.append( ['sg_production_phase','is','Production'] )
            
            sg_filters += filters
            sg_assets  += db.session().find( 'Asset', sg_filters, fields )
        
        # sort the results
        sg_assets.sort( lambda x,y: cmp( x['code'], y['code'] ) )
        
        # generate the entities
        entities    = []
        for i, sg_asset in enumerate(sg_assets):
            entity = Entity( self, self.name(), sg_asset['code'], sg_asset['id'], sg_asset['sg_status_list'] )
            entity.setSortOrder(i)
            entities.append(entity)
        
        return entities
            
    @staticmethod
    def contexts():
        """
        Returns a list of the default contexts in the order that they
        appear for the movie
        
        :return     <list> [ <Context>, .. ]
        """
        keys = AssetContext.ShotgunAssetTypeMap.keys()
        keys.sort()
        return [AssetContext(name) for name in keys]

Context.registerType('Asset',AssetContext)
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

