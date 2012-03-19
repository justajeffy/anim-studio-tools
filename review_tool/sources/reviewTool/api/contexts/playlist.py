##
#   \namespace  reviewTool.contexts.playlist
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       07/27/11
#

import datetime
import os.path
import re

import rv_tools

from .sequence      import SequenceContext

from ..iosystem     import IOSystem
from ..context      import Context
from ..entity       import Entity
from ..version      import Version
from ..clip         import Clip

from ...database    import db
from ...kernel      import core
from ...xml         import XmlDocument

class PlaylistContext(Context):
    def __init__( self, name, data = {} ):
        super(PlaylistContext,self).__init__(name)
        
        # set the custom properties
        self._shotgunId         = data.get('id')
        self._createdAt         = data.get('sg_date_and_time',datetime.datetime.now())
        self._comments          = data.get('description','')
        self._department        = data.get('sg_department',{}).get('name','')
        self._filename          = data.get('filename','')
        
        self._lookupVersions    = data.get('versions',[])
        self._lookupBy          = data.get('lookupBy','id')
    
    def collectEntities( self ):
        self.collectVersions()
        return self.cachedValue('entities')
    
    def collectVersions( self ):
        if ( not self._lookupVersions ):
            self.cache('entities',[])
            self.cache('clips',[])
            self.cache('versions',[])
            return []
        
        # collect all the versions who match the inputed criteria
        sg_versions = []
        lkey        = self._lookupBy
        lookup_versions = [[lkey,'is',version[lkey]] for version in self._lookupVersions]
        
        page = 0
        page_max = 10
        lcount = len(lookup_versions)
        
        while (page < lcount):
            filters     = lookup_versions[page:page+page_max]
            fields      = Version.ShotgunFields + ['entity']
            
            # collect all the shots that are part of this scene from shotgun
            sg_versions += db.session().find('Version',filters,fields,filter_operator='any')
            
            page += page_max
        
        # create a mapping to the versions by their code
        sg_version_map = {}
        for sg_version in sg_versions:
            # ensure that we're only using creative renders when looking up by code
            # since we can have duplicate entries
            if ( lkey == 'code' and not 'creative' in sg_version['sg_tank_address'].lower() ):
                continue
                
            sg_version_map[sg_version[lkey]] = sg_version
        
        entities    = []
        versions    = []
        
        # extract the entity and version information from the clip versions
        for i, lookup_version in enumerate(self._lookupVersions):
            # pull the looked up version based on the key
            sg_version = sg_version_map.get(lookup_version[lkey])
            if ( not sg_version ):
                continue
            
            # retrieve the entity
            sg_entity = sg_version['entity']
            
            # create the entity instance
            entity          = Entity( self, sg_entity['type'], sg_entity['name'], sg_entity['id'] )
            sortOrder       = sg_entity.get('sg_cut_order')
            if ( sortOrder == None ):
                sortOrder = SequenceContext.generateSortOrder(entity.name())
            
            # create the version instance
            lookup_version.update(sg_version)
            version = Version( entity, lookup_version )
            version.setActive(True)
            version.setDefaultPlaylistOrder(i)
            
            # cache the values
            entities.append(entity)
            versions.append(version)
        
        self.cache('entities',entities)
        
        return versions
    
    def comments( self ):
        return self._comments
    
    def createdAt( self ):
        return self._createdAt
    
    def department( self ):
        return self._department
    
    def filename( self ):
        return self._filename
    
    def setShotgunId( self, shotgunId ):
        self._shotgunId = shotgunId
    
    def shotgunId( self ):
        return self._shotgunId
    
    @staticmethod
    def fromFile( filename ):
        filename = str(filename)
        
        # extract the system to be used for loading the inputed file
        ftype = os.path.splitext(filename)[1]
        system = IOSystem.findByType(ftype)
        if ( not system ):
            return None
        
        # load the context from the system
        return system.load(filename)
    
    @staticmethod
    def fromXml( name, xml, filename = '' ):
        """
                Creates a new PlaylistContext instance with the given name by
                loading all relavent data from the xml file
                
                :param      name:
                :type       <str>:
                
                :param      xml:
                :type       <XmlElement>:
                
                :param      filname:
                :type       <str>
                
                :return     <PlaylistContext>:
        """
        # create the new playlist context
        output = PlaylistContext( name, {'filename': filename })
        
        # load the clip data
        xclips          = xml.findChild('clips')
        if ( xclips ):
            entities    = []
            clips       = []
            versions    = []
            
            # load the clip data from the xml file
            for xclip in xclips.children():
                clip    = Clip.fromXml(xclip)
                if ( not clip ):
                    continue
                
                # store the version
                version = clip.version()
                version.setActive(True)
                
                # store the entity
                entity  = clip.entity()
                entity.setContext(output)
                
                # store the clip
                clips.append(clip)
                versions.append(version)
                entities.append(entity)
            
                # store the defaults
                version.setDefaultOptions( clip )
                
            output.cache( 'entities',   entities )
            output.cache( 'clips',      clips )
            output.cache( 'versions',   versions )
        
        return output
        
        
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

