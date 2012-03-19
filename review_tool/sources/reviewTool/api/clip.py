##
#   \namespace  reviewTool.api.clip
#
#   \remarks    Define a Clip class that represents an extension of a version to allow
#               specific playlist overrides
#   
#   \author     Dr. D Studios
#   \date       08/04/11
#

import glob
import os
import re
import rv_tools
import rv_tools.edl

from ..database import db

from .version   import Version
from ..         import settings

class enum(dict):
    def __getattr__( self, key ):
        if ( key in self ):
            return self[key]
        raise AttributeError, key
        
    def __init__( self, *args ):
        count = 1
        for arg in args:
            self[arg] = count
            count *= 2

#--------------------------------------------------------------------------------

class Clip(object):
    # define enum types
    VideoMode = enum('Default','ForcedFrames','ForcedMovie','Custom')
    AudioMode = enum('Default','Published','Custom')
    
    def __eq__( self, other ):
        if ( type(other) == Clip ):
            return self._version == other._version
        elif ( type(other) == Version ):
            return self._version == other
        else:
            return super(Clip,self).__eq__(other)
        
    def __getattr__( self, key ):
        # when no property is found to match,
        # use the base version's key
        return getattr( self._version, key )
        
    def __init__( self, version ):
        self._version           = version
        self._playbackStart     = version.defaultPlaybackStart()
        self._playbackEnd       = version.defaultPlaybackEnd()
        self._playlistOrder     = version.defaultPlaylistOrder()
        
        # define audio properties
        asource                 = version.defaultAudioSource()
        if ( asource ):
            self._audioMode     = Clip.AudioMode.Custom
        elif ( version.defaultAudioOverridden() ):
            asource             = ''
            self._audioMode     = Clip.AudioMode.Published
        else:
            asource             = ''
            self._audioMode     = Clip.AudioMode.Default
        
        self._audioOffset       = version.defaultAudioOffset()
        self._customAudio       = asource
        
        # define video source properties
        vsource = version.defaultVideoSource()
        if ( vsource ):
            self._videoMode     = Clip.VideoMode.Custom
        else:
            vsource             = ''
            self._videoMode     = Clip.VideoMode.Default
            
        self._customVideo       = vsource
        self._customVideoStart = None
        self._customVideoEnd   = None
    
    def audioOverridden( self ):
        """
                Returns whether or not this clip is
                going to use an audio override when
                playing.
                
                :return     <bool>:
        """
        return self.isAudioMode( Clip.AudioMode.Custom | Clip.AudioMode.Published )
    
    def currentAudioSource( self ):
        if ( self.isAudioMode( Clip.AudioMode.Custom ) ):
            return self.customAudio()
        elif ( self.isAudioMode( Clip.AudioMode.Published ) ):
            return self._version.audioSource()
        else:
            return ''
    
    def currentAudioOffset( self ):
        """
                Returns the offset value to be used
                for this clip.  If the clip's audio offset
                value is none, then the base version's offset
                value will be returned.
                
                :return     <int>:
        """
        # return the clip's audio offset
        if ( self._audioOffset != None and self.audioOverridden() ):
            voffset = self._audioOffset
        
        # otherwise, return the version's audio offset
        else:
            voffset = self._version.audioOffset()
        
        # include any offset audio from a custom file
        if ( self.isVideoMode( Clip.VideoMode.Custom ) ):
            voffset += self.sourceStart() - self.playbackStart()
            
        return voffset
        
    def currentVideoSource( self, mode = '' ):
        # use the global source options
        if ( not mode ):
            mode = settings.MEDIA_MODE
        
        if ( self.isVideoMode( Clip.VideoMode.Custom ) ):
            return self._customVideo
        elif ( self.isVideoMode( Clip.VideoMode.ForcedMovie ) ):
            return self._version.videoSource()
        elif ( self.isVideoMode( Clip.VideoMode.ForcedFrames ) ):
            return self._version.imageSource()
        elif ( mode == 'video' ):
            return self._version.videoSource()
        else:
            return self._version.imageSource()
    
    def currentPlaybackStart( self ):
        options = {}
        options['handle']   = self.handleStart()
        options['source']   = self.sourceStart()
        options['cut']      = self.editStart()
        options['custom']   = self.playbackStart()
        
        return options.get(settings.PLAY_MODE,self.playbackStart())
    
    def currentPlaybackEnd( self ):    
        options = {}
        options['handle']   = self.handleEnd()
        options['source']   = self.sourceEnd()
        options['cut']      = self.editEnd()
        options['custom']   = self.playbackEnd()
        return options.get(settings.PLAY_MODE,self.playbackEnd())
    
    def customAudio( self ):
        return self._customAudio
    
    def customVideo( self ):
        return self._customVideo
    
    def displayName( self ):
        entity = self.entity()
        outname = ''
        if ( entity ):
            outname = entity.name()
            
        if ( not outname and self._customVideo ):
            return os.path.basename( self._customVideo )
        return outname
    
    def imageSource( self ):
        """
                Returns the overridden image source
                for this clip if one is provided, otherwise
                will return the original version's image source.
                
                :return     <str>:
        """
        if ( self._videoSource ):
            return self._videoSource
        return self._version.imageSource()
    
    def isAudioMode( self, audioMode ):
        return (self._audioMode & audioMode) != 0
    
    def isVideoMode( self, videoMode ):
        return (self._videoMode & videoMode) != 0
    
    def playbackStart( self ):
        """
                Returns the frame that this clip should playback
                from.
                
                :return     <int>:
        """
        return self._playbackStart
    
    def playbackEnd( self ):
        """
                Returns the frame that this clip should playback
                until.
                
                :return     <int>:
        """
        return self._playbackEnd
        
    def playlistOrder( self ):
        """
                Returns the overridden order for this clip
                to be played in if one is provided, otherwise
                will return the original version's sort order.
                
                :return     <int>:
        """
        if ( self._playlistOrder!= None ):
            return self._playlistOrder
        return self.sortOrder()
    
    def setAudioMode( self, audioMode ):
        self._audioMode = audioMode
    
    def setAudioOffset( self, offset ):
        """
                Overrides the base version's audio offset value
                with the inputed offset value.
                
                :param  offset:
                :type   <int>:
        """
        self._audioOffset = offset
    
    def setAudioOverridden( self, state = True ):
        """
                Sets whether or not this clip is
                going to use an audio override when
                playing.
                
                :param      state:
                :type     <bool>:
        """
        if ( state and self._customAudio ):
            self.setAudioMode( Clip.AudioMode.Custom )
        elif ( state ):
            self.setAudioMode( Clip.AudioMode.Published )
        else:
            self.setAudioMode( Clip.AudioMode.Default )
    
    def setCustomAudio( self, customAudio ):
        """
                Sets the overridden audio source
                for this clip if one is provided, otherwise
                will return the original version's audio source.
                
                :par      audioCustom:
                :type     <str>:
        """
        self._customAudio = customAudio
    
    def setCustomVideo( self, customVideo ):
        customVideo = str(customVideo)
        
        # replace w/ #'s
        results = re.match( '.*\.(\d+)\.[a-z]+$', customVideo)
        if ( results ):
            grp = results.group()
            num = results.groups()[0]
            newgrp = grp.replace( num, '#' * len(num) )
            customVideo = customVideo.replace( grp, newgrp )
        
        self._customVideo = customVideo
        
        # ignore overrides that are not sequences
        if ( not (customVideo and os.path.splitext(customVideo)[1].strip('.') in settings.FILETYPE_SEQUENCE) ):
            self._customVideoStart  = None
            self._customVideoEnd    = None
            return
            
        # grab the source information from the network
        files = glob.glob(customVideo.replace('#','*'))
        if ( not files ):
            self._customVideoStart  = None
            self._customVideoEnd    = None
            return
        
        # generate a new source range information
        files.sort()
        expr    = re.compile( '(\d+)[\.a-zA-Z]+$' )
        start   = int(expr.search(files[0]).groups()[0])
        end     = int(expr.search(files[-1]).groups()[0])
        
        self._customVideoStart   = start
        self._customVideoEnd     = end
    
    def setPlaybackEnd( self, frame ):
        """
                Returns the frame that this clip should playback
                until.
                
                :param      frame:
                :type     <int>:
        """
        self._playbackEnd = frame
    
    def setPlaybackStart( self, frame ):
        """
                Returns the frame that this clip should playback
                until.
                
                :param      frame:
                :type     <int>:
        """
        self._playbackStart = frame
    
    def setPlaylistOrder( self, order ):
        """
                Sets the order this clip should be played
                from the playlist context.
                
                :return     <int>:
        """
        self._playlistOrder = order
    
    def setVideoMode( self, videoMode ):
        self._videoMode = videoMode
    
    def sortKey( self ):
        """
                Returns the default sorting key that will be used
                when ordering clips within the playlist widget.
                
                :return     <str>
        """
        contextOrder    = 10000
        context         = self.context()
        if ( context ):
            contextOrder = context.sortOrder()
        
        return '%08i_%08i_%08i_%s' % (contextOrder,self.sortOrder(),settings.departmentOrder(self.department()),self.name())
    
    def sourceStart( self ):
        if ( self.isVideoMode( Clip.VideoMode.Custom ) and self._customVideoStart != None ):
            return self._customVideoStart
        return self._version.sourceStart()
    
    def sourceEnd( self ):
        if ( self.isVideoMode( Clip.VideoMode.Custom ) and self._customVideoEnd != None ):
            return self._customVideoEnd
        return self._version.sourceEnd()
    
    def toXml( self, xparent ):
        """
                Saves this clip to XML format
                
                :param  xparent:
                :type   <XmlElement>:
        """
        xclip       = xparent.createChild('clip')
        
        # record the clip and its properties
        xclip.setAttribute( 'code', self.code() )
        
        xprops = xclip.createChild('props')
        
        xprops.setProperty( 'playbackStart',    self._playbackStart )
        xprops.setProperty( 'playbackEnd',      self._playbackEnd )
        xprops.setProperty( 'playlistOrder',    self._playlistOrder )
        
        xprops.setProperty( 'audioOffset',      self._audioOffset )
        xprops.setProperty( 'audioMode',        self._audioMode )
        xprops.setProperty( 'customAudio',      self._customAudio )
        xprops.setProperty( 'videoMode',        self._videoMode )
        xprops.setProperty( 'customVideo',      self._customVideo )
        xprops.setProperty( 'customVideoStart', self._customVideoStart )
        xprops.setProperty( 'customVideoEnd',   self._customVideoEnd )
        
        # save the version data
        self._version.toXml( xclip )
        
        return xclip
        
    def version( self ):
        """
                Returns the version instance that this clip represents.
                
                :return     <Version>:
        """
        return self._version
    
    @staticmethod
    def fromData(   clipdata, 
                    defaultDepartment = '', 
                    padLeft = 0,
                    padRight = 0, 
                    overrideAudio = False,
                    active = True ):
        """
                Generates clip information from the inputed list of clip
                data strings
                
                :param      clipdata:
                :type       <list> [ <str>, .. ]
                
                :param      overrideAudio:
                :type       <bool>:
                
                :return     <list> [ <Clip>, .. ]
        """
        from .entity import Entity
        
        clips = []
        
        clipexpr            = re.compile('^(\w+)-?(\w*)(\[([\w]*):?([\w]*)\])?:?(.*)$')
        
        for clipd in clipdata:
            results = clipexpr.match(clipd)
            if ( not results ):
                print 'invalid clip description: ', clipd
                continue
            
            grps         = results.groups()
            base_name    = grps[0]
            base_dept    = grps[1]
            base_version = grps[3]
            base_limit   = grps[4]
            options      = grps[5]
            
            if ( not base_version ):
                base_version = 'latest'
            if ( not base_limit ):
                base_limit = 1
            else:
                base_limit = 1 + int(base_limit)
            
            # extract various clip options
            base_customVideo    = None
            base_customAudio    = None
            base_match          = None
            base_playbackStart  = None
            base_playbackEnd    = None
            base_useAudio       = False
            
            for option in options.split(':'):
                # use the match information
                if ( option.startswith('match_') ):
                    base_match = option
                
                # use path information
                elif ( option.startswith('/') ):
                    if ( base_customVideo != None ):
                        base_customAudio = option
                    else:
                        base_customVideo = option
                
                # use audio information
                elif ( option == 'use_audio' ):
                    base_useAudio = True
                
                # use frame range info
                elif ( re.match( '\d+-\d+', option ) ):
                    base_playbackStart, base_playbackEnd = option.split('-')
                    base_playbackStart  = int(base_playbackStart)
                    base_playbackEnd    = int(base_playbackEnd)
            
            # lookup the shot
            sg_shot = db.session().find_one( 'Shot', [['code','is',base_name]],['code'] )
            if ( not sg_shot ):
                continue
            
            # create the entity
            base = Entity(None,'Shot',sg_shot['code'],sg_shot['id'])
            base.collectCutData()
            
            # load sibling information as well
            prev,next   = base.siblings(padLeft,padRight)
            entities    = prev + [base] + next
            
            for entity in entities:
                # load the versions
                for dept in base_dept.split(','):
                    # create the version filter
                    filters = []
                    filters.append(['sg_tank_address',
                                    'contains',
                                    'SceneShot(%s)' % entity.name()])
                                    
                    filters.append(['sg_tank_address',
                                    'contains',
                                    'Department(%s)' % dept])
                    
                    # load a specified version
                    print 'loading clip:', entity.name(), dept, base_version
                    
                    if ( entity == base and base_version != 'latest' ):
                        filters.append( ['sg_tank_address',
                                         'contains',
                                         'Movie(%s' % base_version] )
                    
                    # sort the results based on the date created
                    order = [{'field_name':'created_at','direction':'desc'}]
                    
                    # look up the creative version from shotgun
                    creative_filt = [['sg_tank_address',
                                     'contains',
                                     'ReviewType(creative)']]
                                       
                    sg_versions = db.session().find('Version', 
                                                     filters + creative_filt, 
                                                     Version.ShotgunFields, 
                                                     order, 
                                                     limit = base_limit)
                    
                    if ( not sg_versions ):
                        sg_versions = db.session().find('Version', 
                                                         filters, 
                                                         Version.ShotgunFields, 
                                                         order, 
                                                         limit = base_limit)
                    
                    if ( not sg_versions ):
                        continue
                    
                    for sg_version in sg_versions:
                        print 'loading version', sg_version['sg_tank_address']
                        # generate the version data
                        v = Version(entity,sg_version)
                        v.collectSourceData()
                        v.setActive(active)
                        
                        # create the clip
                        clip = Clip(v)
                        clip.setAudioOverridden(overrideAudio)
                        
                        # load the audio/video overrides for the base entity
                        if ( entity == base ):
                            if ( base_customVideo ):
                                clip.setCustomVideo( base_customVideo )
                                clip.setVideoMode( Clip.VideoMode.Custom )
                            if ( base_customAudio ):
                                clip.setCustomAudio( base_customAudio )
                            if ( base_useAudio ):
                                clip.setAudioOverridden(True)
                            
                            # set playback information
                            if ( base_match == 'match_source' ):
                                clip.setPlaybackEnd( clip.sourceEnd() )
                                clip.setPlaybackStart( clip.sourceStart() )
                            
                            elif ( base_match == 'match_handle' ):
                                clip.setPlaybackEnd( clip.handleEnd() )
                                clip.setPlaybackStart( clip.handleEnd() )
                            
                            elif ( base_match == 'match_cut' ):
                                pass
                            
                            if ( base_playbackStart and base_playbackEnd ):
                                clip.setPlaybackStart( base_playbackStart )
                                clip.setPlaybackEnd( base_playbackEnd )
                        
                        clips.append(clip)
        
        return clips
        
    @staticmethod
    def fromXml( xclip ):
        """
                Restore the clip data from the xml
                
                :param      xclip:
                :type       <XmlElement>:
                
                :return     <Clip> || None:
        """
        if ( not xclip ):
            return None
        
        version = Version.fromXml( xclip.findChild('version') )
        if ( not version ):
            return None
            
        # create the output clip
        output = Clip( version )
        
        # restore clip properties
        xprops      = xclip.findChild('props')
        
        output._playbackStart     = int(xprops.property('playbackStart',0))
        output._playbackEnd       = int(xprops.property('playbackEnd',0))
        output._playlistOrder     = int(xprops.property('playlistOrder',-1))
        output._audioOffset       = int(xprops.property('audioOffset',0))
        
        # load old style (pre 2.0.9)
        if ( xprops.property('audioOverridden') == 'True' ):
            output._audioMode = Clip.AudioMode.Published
        else:
            output._audioMode = int(xprops.property('audioMode',Clip.AudioMode.Default))
        
        output._customAudio       = xprops.property('customAudio','')
        output._videoMode         = int(xprops.property('videoMode',Clip.VideoMode.Default))
        output._customVideo       = xprops.property('customVideo','')
        output._customVideoStart  = xprops.property('customVideoStart')
        output._customVideoEnd    = xprops.property('customVideoEnd')
        
        if ( output._customVideoStart ):
            output._customVideoStart = int(output._customVideoStart)
        if ( output._customVideoEnd ):
            output._customVideoEnd = int(output._customVideoEnd)
        
        return output
    
    @staticmethod
    def generateRVSequence( clips, mode = None ):
        """
                Generates an RV Edl Sequence class that represents the inputd
                clip instances.  If the use video flag is specified, then the 
                resulting file will be saved out using the video input, otherwise
                the file will be saved out using image sequences.
                
                :param      clips:
                :type       <list> [ <Clip>, .. ]:
                
                :return     <rv_tools.edl.Sequence>:
        """
        # create a new RV sequence
        rv_seq         = rv_tools.edl.Sequence()
        
        for clip in clips:
            options = {}
            
            # make sure the source data is loaded
            clip.collectSourceData()
            
            options['left']                 = str(clip.currentVideoSource(mode))
            options['frame_in']             = clip.currentPlaybackStart()
            options['frame_out']            = clip.currentPlaybackEnd()
            options['frame_min']            = clip.sourceStart()
            options['frame_max']            = clip.sourceEnd()
            options['audio']                = str(clip.currentAudioSource())
            options['use_clip_audio']       = clip.audioOverridden()
            options['audio_offset']         = clip.currentAudioOffset()
            options['audio_frame_start']    = clip.audioStart()
            options['stereo_pair']          = clip.stereoPair()
            
            # create the clip
            rv_clip = rv_seq.add_clip(**options)
            rv_clip.set_meta_data( 'scene_shot',        str(clip.shotName()),        label = 'Shot',             hidden = False )
            rv_clip.set_meta_data( 'department',        str(clip.department()),      label = 'Department',       hidden = False )
            rv_clip.set_meta_data( 'scene_shot_index',  clip.playlistOrder(),                                    hidden = True )
            rv_clip.set_meta_data( 'tank_rev_obj',      str(clip.tankAddress()),     label = 'Tank Revision',    hidden = False )
        
        return rv_seq
    
    @staticmethod
    def playClips( clips, compareMethod = None, mode = None, createSession = True ):
        """
                Launches an RV session with the inputed list of clips to play.
                
                :param      clips:
                :type       <list> [ <Clip>, .. ]:
                
                :param      compareMethod:
                :type       <str>:
                
                :param      mode:
                :type       <str>:
                
                :param      createSession:
                :type       <bool>:
        """
        if ( not clips ):
            return None
            
        # generate the rv sequence
        rv_sequence = Clip.generateRVSequence( clips, mode )
        
        # create the rv session
        session = rv_tools.get_rv_session( new_session = createSession )
        
        # play the sequence in the compare mode
        if ( compareMethod ):
            params = {}
            params['department_meta_id']    = 'department'
            params['scene_shot_meta_id']    = 'scene_shot'
            params['row_order_meta_id']     = 'scene_shot_index'
            params['department_order']      = settings.orderedDepartments()
            
            rv_seq_list = rv_sequence.group_by_department( **params )
            session.compare( rv_seq_list, mode = compareMethod )
        else:
            session.open(rv_sequence)
        
        session.eval('resizeFit()')
        return session
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

