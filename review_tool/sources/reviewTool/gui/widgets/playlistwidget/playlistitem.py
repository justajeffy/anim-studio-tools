##
#   \namespace  reviewTool.gui.widgets.playlistwidget.playlistitem
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

import os.path
import rv_tools.util

from PyQt4.QtCore       import Qt, QVariant, QSize, QTimer, QThread
from PyQt4.QtGui        import QTreeWidgetItem, QPixmap, QLabel, QColor, QPixmapCache

from ..loaderwidget     import LoaderWidget
from ....api.clip       import Clip
from ....               import settings

class ThumbnailThread(QThread):
    def __init__( self ):
        super(ThumbnailThread,self).__init__()
        
        self.sourcefile = ''
        self.targetfile = ''
        self.width      = 150
        self.height     = 40
        
    def run( self ):
        # generate the thumb using rv
        rv_tools.util.generate_thumbnail( self.sourcefile, self.targetfile, self.width, self.height )

#--------------------------------------------------------------------------------

class PlaylistItem( QTreeWidgetItem ):
    def __lt__( self, other ):
        if ( not self._playlist.treeWidget().sortColumn() ):
            morder, mvalid = self.data(  1, Qt.UserRole ).toInt()
            oorder, ovalid = other.data( 1, Qt.UserRole ).toInt()
            
            # sort by playlist order
            if ( morder != -1 and oorder != -1 ):
                return morder < oorder
            
            # group together based on forced sorting
            elif ( morder != -1 or oorder != -1 ):
                return self.text(0) < other.text(0)
            
            # sort by sortkey
            return self.data( 0, Qt.UserRole ).toString() < other.data( 0, Qt.UserRole ).toString()
            
        return super(PlaylistItem,self).__lt__(other)
        
    def __init__( self, playlist, clip ):
        super(PlaylistItem,self).__init__()
        
        # set default properties
        self.setFlags( self.flags() | Qt.ItemIsEditable )
        self.setSizeHint( 0, QSize( 0, 28 ) )
        
        # set custom properties
        self._playlist              = playlist
        self._audioOverridden       = False
        self._clip                  = None
        self._thumbnailThread       = None
        
        # set the clip data
        self.setClip(clip)
        self.setPlaybackStart(  clip.playbackStart())
        self.setPlaybackEnd(    clip.playbackEnd())
        
    def audioOverridden( self ):
        return self._clip.audioOverridden()
    
    def audioOffset( self ):
        return self._clip.audioOffset()
    
    def audioSource( self ):
        return self._clip.audioSource()
    
    def department( self ):
        return self._clip.department()
    
    def clip( self, playlistOrder = -1 ):
        clip = self._clip
        clip.setPlaylistOrder(playlistOrder)
        return clip
    
    def imageSource( self ):
        return self._clip.imageSource()
    
    def loadThumbnail( self ):
        playlist    = self._playlist
        tree        = playlist.treeWidget()
        blocked     = tree.signalsBlocked()
        
        tree.blockSignals(True)
        self.setData( playlist.COLUMNS.index('Preview'), Qt.UserRole, QVariant(self._clip.code()) )
        tree.setItemWidget( self, playlist.COLUMNS.index('Preview'), None )
        tree.blockSignals(blocked)
        
    def playbackEnd( self ):
        return self._clip.playbackEnd()
    
    def playbackStart( self ):
        return self._clip.playbackStart()
    
    def playlistOrder( self ):
        value, success = self.data( 1, Qt.UserRole ).toInt()
        if ( success ):
            return value
        return -1
    
    def setAudioOverridden( self, state = True ):
        self._clip.setAudioOverridden( state )
        
        tree            = self._playlist.treeWidget()
        blocked         = tree.signalsBlocked()
        tree.blockSignals(True)
        self.setText( self._playlist.COLUMNS.index('Audio'), '*' if state else '' )
        tree.blockSignals(blocked)
    
    def setAudioOffset( self, amount ):
        self._clip.setAudioOffset(amount)
    
    def setAudioSource( self, audioSource ):
        self._clip.setAudioSource( audioSource )
    
    def setVideoSource( self, videoSource ):
        self._clip.setVideoSource( videoSource )
    
    def setClip( self, clip ):
        # create a clip for the inputed version
        if ( type(clip) != Clip ):
            clip = Clip(clip)
        
        self._clip  = clip
        playlist    = self._playlist
        tree        = self._playlist.treeWidget()
        
        # collect the source information
        clip.collectSourceData()
        
        editRange       = (clip.editStart(),clip.editEnd())
        sourceRange     = (clip.sourceStart(),clip.sourceEnd())
        handleRange     = (clip.handleStart(),clip.handleEnd())
        blocked         = tree.signalsBlocked()
        tree.blockSignals(True)
        
        # set the user data
        self.setData( playlist.COLUMNS.index('Source'),         Qt.UserRole,    QVariant(clip.sortKey()) )          # used for sorting
        self.setData( playlist.COLUMNS.index('Version'),        Qt.UserRole,    QVariant(clip) )                    # used for the version delegate options
        self.setData( playlist.COLUMNS.index('Custom Start'),   Qt.UserRole,    QVariant(sourceRange) )             # used for the frame delegate range
        self.setData( playlist.COLUMNS.index('Custom End'),     Qt.UserRole,    QVariant(sourceRange) )             # used for the frame delegate range
        
        # set the visual information
        self.setText( playlist.COLUMNS.index('Source'),         clip.displayName() )
        self.setText( playlist.COLUMNS.index('Dept'),           clip.department() )
        self.setText( playlist.COLUMNS.index('Version'),        clip.name() )
        self.setText( playlist.COLUMNS.index('Updated'),        clip.createdAt().strftime('%d/%m/%y %h:%m %a') )
        self.setText( playlist.COLUMNS.index('Artist'),         clip.username() )
        self.setText( playlist.COLUMNS.index('Cut Range'),      ('%i-%i' % editRange) if (editRange[0] or editRange[1]) else '' )
        self.setText( playlist.COLUMNS.index('Handle Range'),   ('%i-%i' % handleRange) if (handleRange[0] or handleRange[1]) else '' )
        self.setText( playlist.COLUMNS.index('Source Range'),   ('%i-%i' % sourceRange) if (sourceRange[0] or sourceRange[1]) else '' )
        
        self.setAudioOverridden( clip.audioOverridden() )
        
        # start the thumbnail generation thread
        iconcache = settings.iconCachePath( clip.code() + '.jpg' )
        if ( not os.path.isfile(iconcache) ):
            tree.setItemWidget( self, playlist.COLUMNS.index('Preview'), LoaderWidget(tree) )
            
            # update the thread data
            self._thumbnailThread = ThumbnailThread()
            self._thumbnailThread.finished.connect( self.loadThumbnail )
            self._thumbnailThread.sourcefile    = clip.videoSource()
            self._thumbnailThread.targetfile    = iconcache
            self._thumbnailThread.start()
        else:
            self.loadThumbnail()
        
        tree.blockSignals(blocked)
        
    def setDepartment( self, department ):
        """
                Sets the department for this playlist item by
                looking for the latest version of the clip's
                entity's versions for the given department.
                
                :return     <bool>: changed
        """
        # update to latest if the versions match
        if ( department == self.department() ):
            return self.updateToLatest()
            
        # find the entity versions
        entity      = self.clip().entity()
        entity.collectVersions()
        versions    = entity.findVersions(department)
        
        # make sure we have some versions to load
        if ( not versions ):
            return False
        
        # set the clip item for this instance to the latest version,
        # preserving the custom playback start/end values
        playbackStart   = self.playbackStart()
        playbackEnd     = self.playbackEnd()
        self.setClip(versions[0])
        self.setPlaybackStart(playbackStart)
        self.setPlaybackEnd(playbackEnd)
        
        return True
    
    def setPlaylistOrder( self, order ):
        tree            = self._playlist.treeWidget()
        blocked         = tree.signalsBlocked()
        tree.blockSignals(True)
        self.setData( 1, Qt.UserRole, QVariant(order) )
        tree.blockSignals(blocked)
    
    def setPlaybackEnd( self, frame ):
        if ( self._clip ):
            self._clip.setPlaybackEnd( frame )
        
        # update the ui
        tree        = self._playlist.treeWidget()
        blocked     = tree.signalsBlocked()
        tree.blockSignals(True)
        self.setData( self._playlist.COLUMNS.index('Custom End'), Qt.EditRole, QVariant(frame) )
        tree.blockSignals(blocked)
    
    def setPlaybackStart( self, frame ):
        if ( self._clip ):
            self._clip.setPlaybackStart( frame )
        
        # update the ui
        tree        = self._playlist.treeWidget()
        blocked     = tree.signalsBlocked()
        tree.blockSignals(True)
        self.setData( self._playlist.COLUMNS.index('Custom Start'), Qt.EditRole, QVariant(frame) )
        tree.blockSignals(blocked)
    
    def setVersionName( self, versionName ):
        """
                Switches the version for this clip to the inputed version name
                
                :return     <bool>: changed
        """
        # find the entity versions
        versions    = self.clip().siblings()
        
        for version in versions:
            if ( versionName == version.name() ):
                self.setClip(version)
                return True
        
        return False
    
    def maximumFrame( self ):
        return self._clip.sourceEnd()
    
    def minimumFrame( self ):
        return self._clip.sourceStart()
    
    def updateToLatest( self ):
        """
                Updates this PlaylistItem to the latest clip version based on
                the current version's department and entity.
                
                :return     <bool>: changed
        """
        # find the entity versions
        self.clip().entity().collectVersions()
        versions    = self.clip().siblings()
        
        # make sure we have some versions to load
        if ( not versions or versions[0] == self.version() ):
            return False
        
        # set the clip item for this instance to the latest version,
        # preserving the custom playback start/end values
        playbackStart   = self.playbackStart()
        playbackEnd     = self.playbackEnd()
        self.setClip(versions[0])
        self.setPlaybackStart(playbackStart)
        self.setPlaybackEnd(playbackEnd)
        
        return True
    
    def version( self ):
        return self._clip.version()
    
    def versionName( self ):
        return self._clip.name()
    
    def videoSource( self ):
        return self._clip.videoSource()
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

