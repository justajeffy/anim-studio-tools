##
#   \namespace  reviewTool.api.context
#
#   \remarks    Defines the main Context class type that will drive the way
#               information is grouped together and out revisions will be
#               shown within context of other revisions
#   
#   \author     Dr. D Studios
#   \date       08/02/11
#

class Context(object):
    _contextTypes   = {}
    
    def __init__( self, name ):
        self._name                  = name
        self._cache                 = {}
        self._versionsLoaded        = False
        self._sortOrder             = -1
    
    def cache( self, key, value ):
        self._cache[str(key)] = value
    
    def cachedValue( self, key, default = None ):
        return self._cache.get(str(key),default)
    
    def clearCache( self ):
        self._cache.clear()
    
    def clips( self ):
        return self._cache.get('clips',[])
    
    def collectEntities( self ):
        return []
    
    def collectCutData( self, entity ):
        return { 'cutStart': 0, 'cutEnd': 0 }
    
    def collectVersions( self ):
        return []
    
    def entities( self ):
        entities = self._cache.get('entities')
        if ( entities == None ):
            entities = self.collectEntities()
            self.cache('entities',entities)
        return entities
    
    def entityVersions( self, entity ):
        self.initVersions()
        return self._cache.get('%s_versions' % entity.name(),{})
    
    def initVersions( self ):
        if ( self._versionsLoaded ):
            return
            
        # collect a list of versions for this context
        versions    = self.versions()
        
        # organize the versions into a cache for this context
        cache       = self.organizeVersions(versions)
        
        # update the cache
        self._cache.update(cache)
    
    def name( self ):
        return self._name
    
    def setSortOrder( self, order ):
        self._sortOrder = order
    
    def siblings( self, entity, padLeft = 1, padRight = 1 ):
        entities = self.entities()
        
        if ( entity in entities ):
            index = entities.index(entity)
            return (entities[index-padLeft:index],entities[index+1:index+padRight+1])
        return ([],[])
    
    def sortOrder( self ):
        return self._sortOrder
    
    def versions( self ):
        versions = self._cache.get('versions')
        if ( versions == None ):
            versions = self.collectVersions()
            self._cache['versions'] = versions
        return versions
    
    def uncache( self, key ):
        key = str(key)
        if ( key in self._cache ):
            return self._cache.pop(key)
        return None
    
    @staticmethod
    def contexts():
        return []
    
    @staticmethod
    def contextType(key):
        return Context._contextTypes.get(str(key))
    
    @staticmethod
    def contextTypeNames():
        keys = Context._contextTypes.keys()
        keys.sort()
        return keys
    
    @staticmethod
    def registerType( name, contextType ):
        Context._contextTypes[str(name)] = contextType
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

