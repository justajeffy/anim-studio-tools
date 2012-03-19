##
#   \namespace  reviewTool.api.entity
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/02/11
#

import socket

from ..database             import db
from ..database.threads     import EntityVersionsThread

from .version   import Version
from .clip      import Clip

class Entity(object):
    def __init__( self, context, entityType, name, shotgunId, status = None ):
        """
                Constructor for the Entity class
                
                :param      context:            the associated context that loaded this entity
                :type       <Context> || None:
                
                :param      entityType:         container type for this entity
                :type       <str>:
                
                :param      name:
                :type       <str>:
                
                :param      shotgunId:
                :type       <int>:
        """
        self._context       = context       # <Context> || None     the context instance this entity is linked to
        self._entityType    = entityType    # <str>                 defines the tank container type for this entity
        self._shotgunId     = shotgunId     # <int>                 id used to lookup shotgun reference
        self._name          = name          # <str>                 name of this entity
        self._status        = status
        
        self._sortOrder     = -1            # <int>                 used to generate sorting keys for clips
        self._cutStart      = None          # <int>                 cut information - defaulted to None to trigger lookup as needed
        self._cutEnd        = None          # <int>                 cut information - defaulted to None to trigger lookup as needed
        self._handleStart   = None          # <int>
        self._handleEnd     = None          # <int>
        self._cache         = {}            # <dict> { <str> key: <variant> value, .. }
        self._versions      = {}            # <dict> { <str> dept: <list> versions [ <Version>, .. ], .. }
    
    def cache( self, key, value ):
        """
                Caches the inputed value on this instance to the given key
                
                :param      key:
                :type       <str>:
                
                :param      value:
                :type       <variant>:
        """
        self._cache[str(key)] = value
    
    def cachedValue( self, key, default = None ):
        """
                Returns the cached value from this instance at the given key
                
                :param      key:
                :type       <str>:
                
                :param      value:
                :type       <variant>:
        """
        return self._cache.get(str(key),default)
    
    def clearCache( self ):
        """
                Clears out the current cache and resets the data for lookup again 
        """
        self._cache.clear()
        self._versions.clear()
        
        self._cutStart      = None
        self._cutEnd        = None
        self._handleStart   = None
        self._handleEnd     = None
    
    def collectVersions( self ):
        """
                Looks up the version information from Shotgun
        """
        if ( self._versions ):
            return
            
        thread = EntityVersionsThread(db.project(),Version.ShotgunFields,self.tankKey())
        thread.run()
        
        self.setVersionsFromShotgun(thread.versions())
    
    def collectCutData( self, attempts = 5 ):
        """
                Collects the cut information for this entity instance from shotgun.
                This method is called internally as needed when the cache needs to be
                reloaded.  To trigger this method again, you should clear the cache using
                clearCache.
                
                :param      attempts:
                :type       <int>:      maximum number of attempts to collect data
        """
        # check to see if the cut data is already loaded
        if ( self._cutEnd != None ):
            return
        
        # create the shotgun query information
        fields  = [ 'sg_cut_start', 'sg_cut_end', 'sg_handle_start', 'sg_handle_end' ]
        filters = [[ 'shot', 'is', {'type':'Shot','id': self.shotgunId()} ]]
        order   = [{'field_name':'id','direction':'desc'}]
        
        # lookup the cut data from shotgun
        sg_cut  = None
        depth   = 0
        while ( not sg_cut and depth < attempts ):
            # wrap the lookupwithin a try/catch to support socket.error's
            try:
                sg_cut = db.session().find_one( 'Cut_sg_shots_Connection', filters, fields, order )
            except socket.error, e:
                pass
                    
            depth += 1
        
        # try to lookup one last time, not blocking the error this time
        if ( not sg_cut ):
            sg_cut = db.session().find_one( 'Cut_sg_shots_Connection', filters, fields, order )
        
        # set the cut data based on the result
        if ( not sg_cut ):
            sg_cut = {}
            
        self._cutStart      = sg_cut.get('sg_cut_start',0)
        self._cutEnd        = sg_cut.get('sg_cut_end',0)
        self._handleStart   = sg_cut.get('sg_handle_start',0)
        self._handleEnd     = sg_cut.get('sg_handle_end',0)
        
    def context( self ):
        """
                Returns the context instance that this entity is associated with
                
                :return     <Context>:
        """
        return self._context
    
    def cutEnd( self ):
        """
                Returns the cut information that is loaded for this entity.  This
                method will automatically call the collectCutData method to lookup
                the data from shotgun if the information is not already cached.
                
                :return     <int>:
        """
        # cache the data
        self.collectCutData()
        
        # return the cut info
        return self._cutEnd
    
    def cutStart( self ):
        """
                Returns the cut information that is loaded for this entity.  This
                method will automatically call the collectCutData method to lookup the
                data from shotgun if the information is not already cached.
                
                :return     <int>:
        """
        self.collectCutData()
        
        return self._cutStart
    
    def entityType( self ):
        """
                Returns the tank entity/container type that this entity represents
                
                :return     <str>:
        """
        return self._entityType
    
    def findVersions( self, department ):
        """
                Looks up versions from this entity's cached version information that
                are associated with the inputed department type.
                
                :param      department:
                :type       <str>:
                
                :return     <list> [ <Version>, .. ]:
        """
        return self._versions.get(str(department),[])
    
    def handleEnd( self ):
        """
                Returns the handle information that is loaded for this entity.  This
                method will automatically call the collectCutData method to lookup
                the data from shotgun if the information is not already cached.
                
                :return     <int>:
        """
        # cache the data
        self.collectCutData()
        
        # return the handle info
        return self._handleEnd
    
    def handleStart( self ):
        """
                Returns the handle information that is loaded for this entity.  This
                method will automatically call the collectCutData method to lookup the
                data from shotgun if the information is not already cached.
                
                :return     <int>:
        """
        self.collectCutData()
        
        return self._handleStart
    
    def name( self ):
        """
                Returns the name for this entity instance
                
                :return     <str>:
        """
        return self._name
    
    def setContext( self, context ):
        self._context = context
    
    def setSortOrder( self, order ):
        """
                Sets the sorting order for this entity in relation to its context
                
                :param      order:
                :type       <int>:
        """
        self._sortOrder = order
    
    def setCutEnd( self, frame ):
        """
                Sets the cut end information for this entity
                
                :param      frame:
                :type       <int>:
        """
        self._cutEnd = frame
    
    def setCutStart( self, frame ):
        """
                Sets the cut start information for this entity
                
                :param      frame:
                :type       <int>:
        """
        self._cutStart = frame
    
    def setVersionsFromShotgun( self, sg_versions ):
        """
                Initializes the version data for this entity to the inputed
                shotgun versions.  This will clear out any existing version
                data, and organize the inputed versions based on their department,
                creating new Version instances for each shotgun version.
                
                :param      sg_versions:
                :type       <list> [ <dict> { <str> key: <variant> value, .. }, .. ]
        """
        # create a list of Version insances from the inputed shotgun versions
        versions = [Version(self,sg_version) for sg_version in sg_versions]
        versions.sort( Version.compare )
        
        # clear out the current version cache
        self._versions = {}
        
        # organize the new versions into dictionaries based on their department
        for version in versions:
            dept = version.department()
            if  ( not dept in self._versions ):
                self._versions[dept] = [version]
            else:
                self._versions[dept].append(version)
    
    def sortOrder( self ):
        """
                Returns the sort order for this entity based on its relation to other
                entities for the given context its in.
                
                :return     <int>:
        """
        return self._sortOrder
    
    def sortVersions( self ):
        """
                Goes through and resorts all the versions based on the current
                sorting criteria
        """
        for versions in self._versions.values():
            versions.sort(Version.compare)
    
    def shotgunId( self ):
        """
                Returns the unique shotgun id pointing to the shotgun record
                for this entity
                
                :return     <int>:
        """
        return self._shotgunId
    
    def siblings( self, padLeft = 1, padRight = 1 ):
        """
                Returns the Entities that are on the left or right for this
                entity based on the inputd padding
                
                :param      padLeft:
                :type       <int>:
                
                :param      padRight:
                :type       <int>:
                
                :return     <tuple> ( <list> [ <Entity>, .. ] previous, <list> [ <Entity>, .. ] following )
        """
        # return the siblings for this entity within its own context
        if ( self.context() ):
            return self.context().siblings( self, padLeft, padRight )
            
        # return the siblings from shotgun based on shot cut order
        if ( self.entityType() == 'Shot' ):
            prev, next = db.findShotSiblings( self.name(), padLeft = padLeft, padRight = padRight )
            
            # create entities from the shtogun information
            prevEntities = [ Entity(None,'Shot',s['code'],s['id'],s['sg_status_list']) for s in prev ]
            nextEntities = [ Entity(None,'Shot',s['code'],s['id'],s['sg_status_list']) for s in next ]
            
            return (prevEntities,nextEntities)
            
        print 'dont know how to look up without context or shot'
        return []
        
    
    def status( self ):
        return self._status
    
    def versions( self ):
        """
                Returns all of the versions that are linked to this entity, broken
                down by the department they're in.
                
                :return     <dict> { <str> dept: <list> [ <Version>, .. ], .. }:
        """
        return self._versions
    
    def uncache( self, key ):
        """
                Removes the inptued key from the current cache on this entity.
                If the key is found, then the cached value is popped off the cache
                and returned.
                
                :param      key:
                :type       <str>:
                
                :return     <variant>:
        """
        key = str(key)
        if ( key in self._cache ):
            return self._cache.pop(key)
        return None
    
    def tankKey( self ):
        """
                Returns the tank key for this entity by combining the entity/container
                type with the name of the entity.
                
                :return     <str>:
        """
        return '%s(%s)' % (self._entityType,self._name)
    
    def toXml( self, xparent ):
        """
                Records this element to the inputed parent's xml element
                
                :param      xparent:
                :type       <XmlElement:
        """
        xentity = xparent.createChild('entity')
        
        # set the attributes
        xentity.setAttribute( 'type',   self._entityType )
        xentity.setAttribute( 'id',     self._shotgunId )
        xentity.setAttribute( 'name',   self._name )
        xentity.setAttribute( 'status', self._status )
        
        # set the properties
        xprops = xentity.createChild('props')
        xprops.setProperty( 'sortOrder',    self._sortOrder )
        xprops.setProperty( 'cutStart',     self._cutStart )
        xprops.setProperty( 'cutEnd',       self._cutEnd )
        xprops.setProperty( 'handleStart',  self._handleStart )
        xprops.setProperty( 'handleEnd',    self._handleEnd )
        
        return xentity
    
    @staticmethod
    def fromXml( xentity, context = None ):
        """
                Restores this entity from the inputed xml
                
                :param      xml:
                :type       <XmlElement>:
                
                :param      context:
                :type       <XmlContext>:
                
                :return     <Entity> || None:
        """
        if ( not xentity ):
            return None
        
        entityType  = xentity.attribute('type','')
        name        = xentity.attribute('name','')
        shotgunId   = int(xentity.attribute('shotgunId',0))
        status      = xentity.attribute('status','')
        
        output = Entity( context, entityType, name, shotgunId, status )
        
        xprops              = xentity.findChild('props')
        output._sortOrder   = int(xprops.property('sortOrder',-1))
        output._cutStart    = int(xprops.property('cutStart',0))
        output._cutEnd      = int(xprops.property('cutEnd',0))
        output._handleStart = int(xprops.property('handleStart',0))
        output._handleEnd   = int(xprops.property('handleEnd',0))
        
        return output
    
    @staticmethod
    def quickLaunch(    clipdata, 
                        playlist = '', 
                        defaultDepartment = '', 
                        padLeft = 0,
                        padRight = 0,
                        overrideAudio = False,
                        compareMethod = None, 
                        mode = None ):
        """
                Launches RV as a quick lookup with the inputed options
                
                :param      clipdata:   space separated list of shot/dept/version codes
                :type       <list> [ <str>, .. ]:
                
                :param      playlist:
                :type       <str>:
                
                :param      defaultDepartment:
                :type       <str> || None:
                
                :param      padLeft:
                :type       <int>:
                
                :param      padRight:
                :type       <int>:
                
                :param      compareMethod:
                :type       <str> || None:
                
                :param      version:
                :type       <str> || None:
                
                :param      overrideAudio:
                :type       <bool>:
                
                :param      mode:
                :type       <str> || None:
                
                :return     <bool> success:
        """
        # make sure we have a valid shot
        if ( not (clipdata or playlist) ):
            print 'you need to provide both a shot and a department to load.'
            return None
        
        clips = []
        
        # load from clip ids
        if ( clipdata ):
            # load the clips from data
            clips = Clip.fromData( clipdata, defaultDepartment, padLeft, padRight, overrideAudio )
        
        # load from playlist
        elif ( playlist ):
            from .contexts.playlist import PlaylistContext
            context = PlaylistContext.fromFile( playlist )
            if ( not context ):
                print 'Could not load playlist context from', playlist
                return None
            
            versions = context.versions()
            for version in versions:
                print 'loading tank data', version.displayName()
                version.collectSourceData()
                clips.append(Clip(version))
        
        print 'playing clips...'
        for clip in clips:
            if ( mode == 'video' ):
                print clip.videoSource()
            else:
                print clip.imageSource()
                
        return Clip.playClips(clips,compareMethod=compareMethod,mode=mode)
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

