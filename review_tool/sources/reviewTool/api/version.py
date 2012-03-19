##
#   \namespace  reviewTool.api.version
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/02/11
#

import datetime
import os
import re
import subprocess
import tank
import time
import webbrowser

from .                  import versiondatacollector

from ..                 import settings
from ..database         import db
from ..kernel           import core

version_name_expr   = re.compile('^(Frames|Movie)\((\d+),')
key_value_expr      = re.compile('(\w+)\((\w+)\)')

#--------------------------------------------------------------------------------

class Version(object):
    # common place to store fields to be looked up for a version instance
    ShotgunFields = [
        'code',
        'id',
        'sg_tank_address',
        'created_at',
        'sg_status_1',
        'sg_comments',
        'user',
    ]
    
    def __eq__( self, other ):
        if ( type(other) == Version ):
            return self._shotgunId == other._shotgunId
        return False
        
    def __init__( self, entity, data, entityNameKey = 'SceneShot' ):
        self._entity        = entity
        self._entityNameKey = entityNameKey
        
        # initialize the custom properties
        self._active        = False
        self._name          = ''
        
        # pop off all None values from the data to properly initialize defaults
        for key, value in data.items():
            if ( value == None ):
                data.pop(key)
        
        # initialize data from shotgun
        self._code          = data.get('code','')
        self._shotgunId     = data.get('id',0)
        self._tankAddress   = data.get('sg_tank_address','')
        self._createdAt     = data.get('created_at',datetime.datetime.now())
        self._status        = data.get('sg_status_1','')
        self._user          = data.get('user',{'type':'HumanUser','name':'','id':0})
        self._comments      = data.get('sg_comments','')
        
        # define the source data information
        self._sourceDataLoaded  = False
        self._sinfoFilepath     = None
        self._stereoPair        = ['left']
        self._videoSource       = ''
        self._imageSource       = ''
        self._audioSource       = ''
        self._sourceStart       = 0
        self._sourceEnd         = 0
        self._audioStart        = 0
        self._audioOffset       = 0
        
        # define default clip info
        self._defaultPlaylistOrder      = data.get('playlistOrder',-1)
        self._defaultAudioOffset        = data.get('audiooffset')
        self._defaultAudioOverridden    = data.get('audioOverridden',False)
        self._defaultAudioSource        = data.get('audioSource')
        
        self._defaultPlaybackStart      = data.get('playbackStart')
        self._defaultPlaybackEnd        = data.get('playbackEnd')
        self._defaultVideoSource        = data.get('videoSource')
        
        # retrieve tank information from the address
        if not self._tankAddress:
            core.warn( 'No valid tank address for: %s' % (self._code) )
            self._tankAddress   = ''
            
        tank_data           = dict(key_value_expr.findall(self._tankAddress))
            
        self._department    = tank_data.get('Department','')
        self._sceneName     = tank_data.get('Scene','')
        self._shotName      = tank_data.get('SceneShot','')
        self._reviewType    = tank_data.get('ReviewType','').lower()
        self._entityName    = tank_data.get(entityNameKey,'')
        
        # initialize information from the data
        results = version_name_expr.match(self._tankAddress)
        if ( results ):
            self._name = results.groups()[1]
        
        # update to reflect the review type
        if ( self._reviewType and self._reviewType != 'creative' ):
            self._name += ' ' + self._reviewType[0].upper()
    
    def audioAddress( self, name = 'master' ):
        options = {}
        options['type']     = settings.SHOT_AUDIO_CONTAINER
        options['name']     = name
        options['scene']    = self.sceneName()
        options['shot']     = self.shotName()
        
        return 'Audio(latest, %(type)s(%(name)s, Scene(%(scene)s), SceneShot(%(shot)s)))' % options
    
    def audioOffset( self ):
        return self._audioOffset
    
    def audioSource( self ):
        return self._audioSource
    
    def audioStart( self ):
        return self._audioStart
    
    def code( self ):
        return self._code
    
    def collectAncestors( self, maxDepth = 20 ):
        """
                Collects the ancestors of this version and returns them
                
                :param  maxDepth:
                :type   <int>:
                
                :return     <list> [ <TankObject>, .. ]:
        """
        
        def _worker( tank_obj, depth ):
            if ( not tank_obj or (maxDepth and depth == maxDepth) ):
                return []
            
            output  = [tank_obj]
            
            for dep in tank_obj.dependencies.get_value():
                output += _worker(dep,depth+1)
            return output
        
        return _worker( self.tankObject(), 0 )
    
    def collectSourceData( self ):
        # check to see if the data has already been loaded
        if ( self._sourceDataLoaded ):
            return
        
        self._sourceDataLoaded = True
        
        if ( self.isNull() ):
            return
        
        imageSource, sourceStart, sourceEnd, stereo_pair    = versiondatacollector.collectFrameData( self.tankAddress() )
        videoSource                                         = versiondatacollector.collectMovieData( self.tankAddress() )
        audioSource, audioStart, audioOffset                = versiondatacollector.collectAudioData( self.audioAddress() )
        
        self._imageSource   = imageSource
        self._videoSource   = videoSource
        self._audioSource   = audioSource
        self._sourceStart   = sourceStart
        self._sourceEnd     = sourceEnd
        self._audioStart    = audioStart
        self._audioOffset   = audioOffset
        self._stereoPair    = stereo_pair
        
    def comments( self ):
        return self._comments
    
    def context( self ):
        entity = self.entity()
        if ( entity ):
            return entity.context()
        return None
    
    def createdAt( self ):
        return self._createdAt
    
    def defaultAudioOverridden( self ):
        return self._defaultAudioOverridden
        
    def defaultAudioSource( self ):
        return self._defaultAudioSource
        
    def defaultAudioOffset( self ):
        return self._defaultAudioOffset
    
    def defaultPlaybackEnd( self ):
        if ( self._defaultPlaybackEnd != None ):
            return self._defaultPlaybackEnd
            
        estart  = self.editStart()
        eend    = self.editEnd()
        if ( estart or eend ):
            return eend
        
        return self.sourceEnd()
    
    def defaultPlaylistOrder( self ):
        return self._defaultPlaylistOrder
    
    def defaultPlaybackStart( self ):
        if ( self._defaultPlaybackStart != None ):
            return self._defaultPlaybackStart
            
        estart  = self.editStart()
        eend    = self.editEnd()
        if ( estart or eend ):
            return estart
        
        return self.sourceStart()
    
    def defaultVideoSource( self ):
        return self._defaultVideoSource
    
    def defineSourceData( self, data ):
        
        self._sourceDataLoaded  = True
        self._imageSource       = data.get('imageSource','')
        self._videoSource       = data.get('videoSource','')
        self._sourceStart       = data.get('sourceStart',0)
        self._sourceEnd         = data.get('sourceEnd',0)
        self._audioSource       = data.get('audioSource','')
        self._audioOffset       = data.get('audioOffset',0)
        self._audioStart        = data.get('audioStart',0)
        
        return True
    
    def department( self ):
        return self._department
    
    def displayName( self ):
        """
                Returns the display name for this version
                
                :return     <str>:
        """
        return '%s %s (%s v%s)' % (self.shotName(),self.department(),self.reviewType(),self.name())
    
    def editEnd( self ):
        entity = self.entity()
        if ( entity ):
            return entity.cutEnd()
        return 0
    
    def editStart( self ):
        entity = self.entity()
        if ( entity ):
            return entity.cutStart()
        return 0
    
    def entity( self ):
        return self._entity
    
    def entityName( self ):
        return self._entityName
    
    def entityNameKey( self ):
        return self._entityNameKey
    
    def explore( self ):
        sid = self.shotgunId()
        if ( sid ):
            webbrowser.open('http://shotgun/detail/Version/%s' % sid)
        return sid
    
    def handleEnd( self ):
        entity = self.entity()
        if ( entity ):
            return entity.handleEnd()
        return 0
    
    def handleStart( self ):
        entity = self.entity()
        if ( entity ):
            return entity.handleStart()
        return 0
    
    def imageSource( self ):
        return self._imageSource
    
    def isActive( self ):
        return self._active
    
    def isNull( self ):
        return self._code == ''
    
    def user( self ):
        return self._user
    
    def username( self ):
        return self._user.get('name','')
    
    def name( self ):
        return self._name
    
    def reviewType( self ):
        return self._reviewType
    
    def sceneName( self ):
        return self._sceneName
    
    def shotName( self ):
        return self._shotName
    
    def shotgunId( self ):
        return self._shotgunId
    
    def sinfoFilepath( self ):
        if ( self._sinfoFilepath != None ):
            return self._sinfoFilepath
        
        # extract the sinfo address
        sinfo_address = self.tankAddress().replace('Frames','SInfoFile').replace('Movie','SInfoFile')
        if ( not sinfo_address ):
            self._sinfoFilepath = ''
            return self._sinfoFilepath
        
        try:
            self._sinfoFilepath = tank.find(sinfo_address).system.filesystem_location
        except:
            self._sinfoFilepath = ''
            
        return self._sinfoFilepath
    
    def siblingsOfCommonDescent( self ):
        """
                Looks up all of the siblings for this version whose ancestry is shared
        """
        ancestors   = set(self.collectAncestors())
        siblings    = [ sibling for sibling in self.siblings() if sibling != self and ancestors.intersection(sibling.collectAncestors()) ]
        return siblings
    
    def siblings( self ):
        """
                Returns an ordered list of the versions that are linked to
                this version based on its entity
                
                :return     <list> [ <Version>, .. ]:
        """
        entity = self.entity()
        if ( entity ):
            return entity.findVersions( self.department() )
        return []
    
    def siblingNames( self ):
        """
                Returns a list of the names for the versions that are
                linked to this version based on its entity.
                
                :return     <list> [ <str>, .. ]
        """
        return [ version.name() for version in self.siblings() ]
        
    def setActive( self, state = True ):
        self._active = state
    
    def setDefaultOptions( self, clip ):
        self._defaultPlaylistOrder      = clip._playlistOrder
        
        self._defaultAudioOffset        = clip._audioOffset
        self._defaultAudioOverridden    = clip.audioOverridden()
        self._defaultAudioSource        = clip._audioSource
        
        self._defaultPlaybackStart      = clip._playbackStart
        self._defaultPlaybackEnd        = clip._playbackEnd
        self._defaultVideoSource        = clip._videoSource
    
    def setDefaultPlaylistOrder( self, order ):
        self._defaultPlaylistOrder = order
    
    def sourceDataLoaded( self ):
        return self._sourceDataLoaded
    
    def sourceEnd( self ):
        return self._sourceEnd
    
    def sourceStart( self ):
        return self._sourceStart
    
    def sortOrder( self ):
        entity = self.entity()
        if ( entity ):
            return entity.sortOrder()
        return 0
    
    def status( self ):
        return self._status
    
    def stereoPair( self ):
        return self._stereoPair
    
    def swapEyes( self ):
        """
                Determines whether or not this clip should swap eyes
                when rendering based on what the dominient eye is
             
                :return <bool>:
        """
        return self._stereoPair[0] == 'right'
    
    def tankAddress( self ):
        return self._tankAddress
    
    def tankObject( self ):
        if ( self._tankAddress ):
            return tank.find(self._tankAddress.replace('Movie(','Frames(')).get_object()
        return None
    
    def toXml( self, xparent ):
        # create the version xml
        xversion = xparent.createChild('version')
        xversion.setAttribute( 'name', self._name )
        xversion.setAttribute( 'code', self._code )
        
        # save shotgun info
        xshotgun = xversion.createChild( 'shotgun' )
        
        xshotgun.setAttribute( 'id',        self._shotgunId )
        xshotgun.setAttribute( 'address',   self._tankAddress )
        xshotgun.setAttribute( 'createdAt', time.mktime(self._createdAt.timetuple()) )
        xshotgun.setAttribute( 'status',    self._status )
        xshotgun.setProperty( 'comments',   self._comments )
        
        xuser = xshotgun.createChild('user')
        for key, value in self._user.items():
            xuser.setAttribute( key, value )
        
        # set additional information
        xprops = xversion.createChild('props')
        
        xprops.setProperty( 'stereoPair',        ':'.join(self._stereoPair) )
        xprops.setProperty( 'videoSource',      self._videoSource )
        xprops.setProperty( 'imageSource',      self._imageSource )
        xprops.setProperty( 'audioSource',      self._audioSource )
        xprops.setProperty( 'sourceStart',      self._sourceStart )
        xprops.setProperty( 'sourceEnd',        self._sourceEnd )
        xprops.setProperty( 'audioStart',       self._audioStart )
        xprops.setProperty( 'audioOffset',      self._audioOffset )
        xprops.setProperty( 'department',       self._department )
        xprops.setProperty( 'scene',            self._sceneName )
        xprops.setProperty( 'shot',             self._shotName )
        xprops.setProperty( 'reviewType',       self._reviewType )
        xprops.setProperty( 'entityName',       self._entityName )
        
        # store the entity
        self._entity.toXml( xversion )
        
        return xversion
        
    def videoSource( self ):
        return self._videoSource
    
    @staticmethod
    def compare( a, b ):
        # compare based on review types
        if  ( a._reviewType != b._reviewType ):
            return cmp(settings.REVIEW_TYPE_ORDER.index(a._reviewType),settings.REVIEW_TYPE_ORDER.index(b._reviewType))
            
        # compare versions by status first
        if ( not settings.SORT_VERSIONS_BY_DATE and (a._status != b._status) ):
            return cmp(settings.versionOrder(a._status),settings.versionOrder(b._status))
        
        # compare by date
        return cmp(b._createdAt,a._createdAt)
    
    @staticmethod
    def findCode( filename ):
        """
                Extracts the unique code for a version given the inputed filename
                information.
                
                :param      filename:
                :type       <str>:
                
                :return     <str>: blank string will be returned if not found
        """
        results = re.match( '(r\d+_)?(.*)', os.path.basename(str(filename)).split('.')[0] )
        if ( results ):
            return results.groups()[1]
        return ''
    
    @staticmethod
    def fromXml( xversion ):
        """
                Restore the version data from the xml
                
                :param      xclip:
                :type       <XmlElement>:
                
                :return     <Version> || None:
        """
        if ( not xversion ):
            return None
            
        # restore the shotgun data
        from .entity import Entity
        entity = Entity.fromXml( xversion.findChild('entity') )
        if ( not entity ):
            return None
        
        output = Version( entity, {}, xversion.attribute('entityNameKey') )
        
        # restore properties
        output._name    = xversion.attribute('name')
        output._code    = xversion.attribute('code')
        
        # restore shotgun properties
        xshotgun            = xversion.findChild('shotgun')
        output._shotgunId   = int( xshotgun.attribute('id',0) )
        output._tankAddress = xshotgun.attribute('address','')
        output._createdAt   = datetime.datetime.fromtimestamp(float(xshotgun.attribute('createdAt',0)))
        output._status      = xshotgun.attribute('status','')
        output._comments    = xshotgun.property('comments','')
        
        xuser               = xshotgun.findChild('user')
        attrs               = {}
        attrs['type']       = xuser.attribute('type','')
        attrs['name']       = xuser.attribute('name','')
        attrs['id']         = int(xuser.attribute('id',0))
        output._user        = attrs
        
        # restore custom properties
        xprops                      = xversion.findChild('props')
        output._sourceDataLoaded    = True
        output._stereoPair          = xprops.property('stereoPair','left').split(':')
        output._videoSource         = xprops.property('videoSource','')
        output._imageSource         = xprops.property('imageSource','')
        output._audioSource         = xprops.property('audioSource','')
        output._sourceStart         = int(xprops.property('sourceStart',0))
        output._sourceEnd           = int(xprops.property('sourceEnd',0))
        output._audioStart          = int(xprops.property('audioStart',0))
        output._audioOffset         = int(xprops.property('audioOffset',0))
        output._department          = xprops.property('department','')
        output._sceneName           = xprops.property('scene','')
        output._shotName            = xprops.property('shot','')
        output._reviewType          = xprops.property('reviewType','')
        output._entityName          = xprops.property('entityName','')
        
        return output
        
    @staticmethod
    def fromFile( filename, context = None ):
        """
                Loads a version instance from the inputed filename by looking up its
                data using the code from the filename.  If the data cannot be retrieved
                from shotgun, a None value will be returned.
                
                :param      filename:
                :type       <str>:
                
                :param      context:
                :type       <Context>:
                
                :return     <Version> || None:
        """
        # make sure we have a valid code
        code = Version.findCode(filename)
        if ( not code ):
            return None
        
        # look up the version based on its code
        sg_version = db.session().find_one( 'Version', [['code','is',code]], Version.ShotgunFields + ['entity'] )
        if ( not sg_version ):
            return None
            
        # create an entity
        sg_entity   = sg_version['entity']
        
        # import entity within this method to avoid dependent imports
        from .entity            import Entity
        entity      = Entity( context, sg_entity['type'], sg_entity['name'], sg_entity['id'] )
        
        # create the version instance
        version = Version( entity, sg_version )
        
        return version
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

