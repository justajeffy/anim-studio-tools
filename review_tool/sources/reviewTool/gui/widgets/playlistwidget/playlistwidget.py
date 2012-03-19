##
#   \namespace  reviewTool.gui.widgets.playlistwidget.playlistwidget
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

import os.path

import PyQt4.uic

from PyQt4.QtCore                   import  pyqtSignal,\
                                            Qt
                                            
from PyQt4.QtGui                    import  QApplication,\
                                            QCursor,\
                                            QMenu,\
                                            QFileDialog,\
                                            QInputDialog,\
                                            QWidget

from ...delegates.griddelegate      import  GridDelegate
from ....api.clip                   import Clip
from ....kernel                     import  core
from ....                           import  settings
from ....                           import  resources

from .playlistitem                  import  PlaylistItem
from .playlisttree                  import  PlaylistTree        # will be auto-assigned to uiPlaylistTREE via uic promotion

from .delegates.framedelegate       import  FrameDelegate
from .delegates.deptdelegate        import  DeptDelegate
from .delegates.versiondelegate     import  VersionDelegate
from .delegates.thumbnaildelegate   import  ThumbnailDelegate

class PlaylistWidget( QWidget ):
    selectionChanged        = pyqtSignal()
    versionLookupRequested  = pyqtSignal(str)
    playlistChanged         = pyqtSignal()
    
    COLUMNS = [
        'Source',
        'Dept',
        'Audio',
        'Version',
        'Custom Start',
        'Custom End',
        'Cut Range',
        'Handle Range',
        'Source Range',
        'Updated',
        'Artist',
        'Preview'
    ]
    
    BLANK_LABELS = [
        'Audio',
        'Preview'
    ]
    
    HIDDEN_COLUMNS = []     # driven by user preferences, adding to this list
                            # here will do nothing
    
    def __init__( self, parent ):
        # initialize the super class
        QWidget.__init__( self, parent )
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'ui/playlistwidget.ui' )
        PyQt4.uic.loadUi( uifile, self )
        
        # toggle the editor off
        self.uiEditBTN.setChecked(False)
        
        # create custom properties
        self._useVideo              = True
        self._context               = None
        self._popupMenu             = None
        self._currentHeaderColumn   = None      # used during header menu popup
        
        # create the delegate
        self.uiPlaylistTREE.setItemDelegate(GridDelegate(self.uiPlaylistTREE))
        self.uiPlaylistTREE.setColumnCount( len( self.COLUMNS ) )
        self.uiPlaylistTREE.setHeaderLabels( [ '' if key in self.BLANK_LABELS else key for key in self.COLUMNS ] )
        self.uiPlaylistTREE.setEditableColumns( [ PlaylistWidget.COLUMNS.index(column) for column in ('Dept','Version','Custom Start','Custom End') ] )
        
        for col in self.HIDDEN_COLUMNS:
            self.uiPlaylistTREE.setColumnHidden( self.COLUMNS.index(col), True )
        
        # initialize the header
        header = self.uiPlaylistTREE.header()
        hitem  = self.uiPlaylistTREE.headerItem()
        
        header.setContextMenuPolicy( Qt.CustomContextMenu )
        header.setResizeMode(0,header.Stretch)
        header.setResizeMode(self.COLUMNS.index('Audio'), header.ResizeToContents)
        
        # setup the delegates
        deptdelegate        = DeptDelegate(         self.uiPlaylistTREE )
        framedelegate       = FrameDelegate(        self.uiPlaylistTREE )
        versiondelegate     = VersionDelegate(      self.uiPlaylistTREE )
        thumbnaildelegate   = ThumbnailDelegate(    self.uiPlaylistTREE )
        
        self.uiPlaylistTREE.setItemDelegateForColumn( self.COLUMNS.index('Dept'),           deptdelegate )
        self.uiPlaylistTREE.setItemDelegateForColumn( self.COLUMNS.index('Version'),        versiondelegate )
        self.uiPlaylistTREE.setItemDelegateForColumn( self.COLUMNS.index('Custom Start'),   framedelegate )
        self.uiPlaylistTREE.setItemDelegateForColumn( self.COLUMNS.index('Custom End'),     framedelegate )
        self.uiPlaylistTREE.setItemDelegateForColumn( self.COLUMNS.index('Preview'),        thumbnaildelegate )
        
        # create the icons
        self.uiSelectAllBTN.setIcon(    resources.icon( 'img/playlist/select_all.png' ) )
        self.uiOptionsBTN.setIcon(      resources.icon( 'img/playlist/options.png' ) )
        self.uiEditBTN.setIcon(         resources.icon( 'img/playlist/edit.png' ) )
        self.uiMoveUpBTN.setIcon(       resources.icon( 'img/playlist/move_up.png' ) )
        self.uiMoveDownBTN.setIcon(     resources.icon( 'img/playlist/move_down.png' ) )
        self.uiCloseEditorBTN.setIcon(  resources.icon( 'img/playlist/close.png' ) )
        
        hitem.setIcon( self.COLUMNS.index('Audio'),     resources.icon( 'img/playlist/audio.png' ) )
        hitem.setIcon( self.COLUMNS.index('Dept'),      resources.icon( 'img/main/filter_departments.png' ) )
        hitem.setIcon( self.COLUMNS.index('Cut Range'), resources.icon( 'img/playlist/cut.png' ) )
        
        # create connections
        self.uiPlaylistTREE.itemChanged.connect(                self.updateItem )
        self.uiPlaylistTREE.itemDoubleClicked.connect(          self.lookupItem )
        self.uiPlaylistTREE.itemSelectionChanged.connect(       self.emitSelectionChanged )
        self.uiPlaylistTREE.itemSelectionChanged.connect(       self.refreshAdvanced )
        self.uiPlaylistTREE.customContextMenuRequested.connect( self.showPopupMenu )
        
        self.uiSelectAllBTN.clicked.connect(                self.selectAll )
        self.uiMoveUpBTN.clicked.connect(                   self.moveUp )
        self.uiMoveDownBTN.clicked.connect(                 self.moveDown )
        
        # connect the edit menu
        self.uiVideoDefaultCHK.clicked.connect(             self.commitEditData )
        self.uiVideoFramesCHK.clicked.connect(              self.commitEditData )
        self.uiVideoMovieCHK.clicked.connect(               self.commitEditData )
        self.uiVideoCustomCHK.clicked.connect(              self.commitEditData )
        self.uiAudioDefaultCHK.clicked.connect(             self.commitEditData )
        self.uiAudioPublishedCHK.clicked.connect(           self.commitEditData )
        self.uiAudioCustomCHK.clicked.connect(              self.commitEditData )
        self.uiPlaybackEndSPN.editingFinished.connect(      self.commitEditData )
        self.uiPlaybackStartSPN.editingFinished.connect(    self.commitEditData )
        self.uiVideoTXT.textEdited.connect(                 self.commitCustomData )
        self.uiAudioTXT.textEdited.connect(                 self.commitCustomData )
        self.uiVideoBTN.clicked.connect(                    self.pickVideo )
        self.uiAudioBTN.clicked.connect(                    self.pickAudio )
        
        header.customContextMenuRequested.connect(          self.showHeaderMenu )
        
        self.uiPlaylistTREE.sortByColumn( 0, Qt.AscendingOrder )
        self.refreshAdvanced()
    
    def blockUi( self, state ):
        self.uiPlaylistTREE.blockSignals(state)
        self.uiPlaylistTREE.setUpdatesEnabled(not state)
    
    def commitCustomData( self ):
        # make sure we have an item
        item = self.uiPlaylistTREE.currentItem()
        if ( not item ):
            return False
        
        # update the video information
        clip = item.clip()
        clip.setCustomVideo( self.uiVideoTXT.text() )
        clip.setCustomAudio( self.uiAudioTXT.text() )
    
        # refresh the item in the ui
        item.setClip(clip)
        
    def commitEditData( self ):
        """
                Updates the current tree item with the latest editing information
                from the edit panel.
                
                :return     <bool>: success
        """
        # make sure we have an item
        item = self.uiPlaylistTREE.currentItem()
        if ( not item ):
            return False
        
        # update the video information
        clip = item.clip()
        if ( self.uiVideoCustomCHK.isChecked() ):
            clip.setVideoMode( clip.VideoMode.Custom )
        elif ( self.uiVideoFramesCHK.isChecked() ):
            clip.setVideoMode( clip.VideoMode.ForcedFrames )
        elif ( self.uiVideoMovieCHK.isChecked() ):
            clip.setVideoMode( clip.VideoMode.ForcedMovie )
        else:
            clip.setVideoMode( clip.VideoMode.Default )
        
        # refresh the ui
        self.uiVideoTXT.blockSignals(True)
        self.uiVideoTXT.setText( clip.currentVideoSource() )
        self.uiVideoTXT.blockSignals(False)
        
        # update the audio information
        if ( self.uiAudioCustomCHK.isChecked() ):
            clip.setAudioMode( clip.AudioMode.Custom )
        elif ( self.uiAudioPublishedCHK.isChecked() ):
            clip.setAudioMode( clip.AudioMode.Published )
        else:
            clip.setAudioMode( clip.AudioMode.Default )
        
        self.uiAudioTXT.blockSignals(True)
        self.uiAudioTXT.setText( clip.currentAudioSource() )
        self.uiAudioTXT.blockSignals(False)
        
        clip.setAudioOffset( self.uiAudioOffsetSPN.value() )
        
        # update frame range info
        item.setPlaybackStart(  self.uiPlaybackStartSPN.value() )
        item.setPlaybackEnd(    self.uiPlaybackEndSPN.value() )
        
        # update the hints
        self.uiSourceStartLBL.setText(  'Min: %04i' % clip.sourceStart() )
        self.uiSourceEndLBL.setText(    'Max: %04i' % clip.sourceEnd() )
        
        # refresh the item in the ui
        item.setClip(clip)
        
        return True
    
    def context( self ):
        return self._context
    
    def clips( self ):
        tree    = self.uiPlaylistTREE
        return [ tree.topLevelItem(i).clip(i) for i in range( tree.topLevelItemCount() ) ]
        
    def currentItem( self ):
        items = self.uiPlaylistTREE.selectedItems()
        if ( len(items) == 1 ):
            return items[0]
        return None
    
    def emitSelectionChanged( self ):
        """
                Emits the selection changed signal, provided the signals
                are not currently being blocked for this widget
        """
        if ( not self.signalsBlocked() ):
            self.selectionChanged.emit()
    
    def emitPlaylistChanged( self ):
        """
                Emits the playlist changed signal, provided the signals
                are not currently being blocked for this widget
        """
        if ( not self.signalsBlocked() ):
            self.playlistChanged.emit()
    
    def emitVersionLookupRequested( self, code ):
        if ( not self.signalsBlocked() ):
            self.versionLookupRequested.emit(code)
    
    def fixCustomRanges( self, items ):
        self.blockUi(True)
        
        for item in items:
            sstart  = item.clip().sourceStart()
            send    = item.clip().sourceEnd()
            
            if ( not ( sstart <= item.playbackStart() and item.playbackStart() <= send ) ):
                item.setPlaybackStart(sstart)
            if ( not ( sstart <= item.playbackEnd() and item.playbackEnd() <= send ) ):
                item.setPlaybackEnd(send)
                
        self.blockUi(False)
    
    def focusInEvent( self, event ):
        super(PlaylistWidget,self).focusInEvent(event)
        self.uiPlaylistTREE.setFocus()
    
    def lookupItem( self, item ):
        self.emitVersionLookupRequested(item.clip().code())
    
    def isEmpty( self ):
        return self.uiPlaylistTREE.topLevelItemCount() == 0
    
    def overrideAudio( self, state = True, dept = None ):
        for item in self.uiPlaylistTREE.selectedItems():
            if ( dept == None or item.clip().department() == dept ):
                item.setAudioOverridden(state)
    
    def moveUp( self ):
        self.moveItems( self.selectedItems(), -1 )
    
    def matchCustomRanges( self, items, matchTo ):
        self.blockUi(True)
        
        for item in items:
            # update to cut info
            if ( matchTo == 'edit' ):
                item.setPlaybackStart(  item.clip().editStart())
                item.setPlaybackEnd(    item.clip().editEnd())
            
            # update to source info
            elif ( matchTo == 'source' ):
                item.setPlaybackStart(  item.clip().sourceStart())
                item.setPlaybackEnd(    item.clip().sourceEnd())
                
            # update to handle info
            elif ( matchTo == 'handle' ):
                item.setPlaybackStart(  item.clip().handleStart())
                item.setPlaybackEnd(    item.clip().handleEnd())
                
        self.blockUi(False)
    
    def moveDown( self ):
        items = self.selectedItems()
        items.reverse()
        self.moveItems( items, 1 )
    
    def moveItems( self, items, spaces ):
        """
                Moves the inputed items the number of spaces up or down
                
                :param      items:
                :type       <list> [ <QTreeWidgetItem>, .. ]:
                
                :param      spaces:
                :type       <int>
                
                :return     <bool> altered
        """
        altered = False
        for item in items:
            if ( self.moveItem( item, spaces ) ):
                altered = True
        return altered
    
    def moveItem( self, item, spaces ):
        """
                Moves the inputed tree item the given number of spaces up or down
                along the tree
                
                :param      item:
                :type       <QTreeWidgetItem>:
                
                :param      spaces:
                :type       <int>:
                
                :return     <bool>: success
        """
        # make sure we have a valid item
        if ( not item ):
            return False
        
        # make sure we are moving to a valid location
        tree        = self.uiPlaylistTREE
        items       = [ tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) ]
        count       = len(items)
        index       = items.index(item)
        new_index   = index + spaces
        
        if ( new_index < 0 ):
            new_index = 0
        elif ( new_index >= count ):
            new_index = count - 1
        
        # make sure we're actually needing to move somewhere
        if ( new_index == index ):
            return False
            
        # move the item around in the tree
        self.blockUi(True)
        tree.setSortingEnabled(False)
        
        items.remove(item)
        items.insert(new_index,item)
        for i in range(count):
            items[i].setPlaylistOrder(i)
            
        self.blockUi(False)
        tree.setSortingEnabled(True)
        
        return True
    
    def popupMenu( self ):
        return self._popupMenu
    
    def pickVideo( self ):
        """
                Prompts the user to select a new video source
                for the current playlist item.
                
                :return     <bool>: success
        """
        filename = QFileDialog.getOpenFileName( self, 'Select Video Source', self.uiVideoTXT.text(), settings.videoFileTypes() )
        if ( filename ):
            self.uiVideoSourceTXT.setText(filename) 
            return True
        return False
    
    def pickAudio( self ):
        """
                Prompts the user to select a new audio source
                for the current playlist item.
                
                :return     <bool>: success
        """
        filename = QFileDialog.getOpenFileName( self, 'Select Audio Source', self.uiAudioTXT.text(), settings.audioFileTypes() )
        if ( filename ):
            self.uiAudioTXT.setText(filename)
            return True
        return False
    
    def removeVersions( self, versions ):
        self.blockUi(True)
        
        for i in range( self.uiPlaylistTREE.topLevelItemCount() - 1, -1, -1 ):
            item = self.uiPlaylistTREE.topLevelItem(i)
            if ( not item.version() in versions ):
                continue
            
            self.uiPlaylistTREE.takeTopLevelItem(i)
            item.version().setActive(False)
        
        self.blockUi(False)
        
        self.refreshAdvanced()
    
    def refreshAdvanced( self ):
        # determine the enabled state
        enabled = len(self.uiPlaylistTREE.selectedItems()) > 0
        
        # update the enabled state
        self.uiEditBTN.setEnabled(enabled)
        self.uiMoveUpBTN.setEnabled(enabled)
        self.uiMoveDownBTN.setEnabled(enabled)
        
        item = self.currentItem()
        if ( not item ):
            self.uiEditorFRAME.setEnabled(False)
            self.uiVideoTXT.setText('')
            self.uiAudioTXT.setText('')
            self.uiSourceEndLBL.setText(    'Min: 0000' )
            self.uiSourceStartLBL.setText(  'Max: 0000' )
            return
        
        self.uiEditorFRAME.setEnabled(True)
        clip = item.clip()
        
        # update the video source mode
        if ( clip.isVideoMode( clip.VideoMode.Custom ) ):
            self.uiVideoCustomCHK.setChecked(True)
        elif ( clip.isVideoMode( clip.VideoMode.ForcedFrames ) ):
            self.uiVideoFramesCHK.setChecked(True)
        elif ( clip.isVideoMode( clip.VideoMode.ForcedMovie ) ):
            self.uiVideoMovieCHK.setChecked(True)
        else:
            self.uiVideoDefaultCHK.setChecked(True)
        
        # update video text
        self.uiVideoTXT.blockSignals(True)
        self.uiVideoTXT.setText( clip.currentVideoSource() )
        self.uiAudioTXT.blockSignals(False)
        
        # update the audio source information
        if ( clip.isAudioMode( clip.AudioMode.Custom ) ):
            self.uiAudioCustomCHK.setChecked(True)
        elif ( clip.isAudioMode( clip.AudioMode.Published ) ):
            self.uiAudioPublishedCHK.setChecked(True)
        else:
            self.uiAudioDefaultCHK.setChecked(True)
        
        # update audio offset data
        self.uiAudioOffsetSPN.blockSignals(True)
        self.uiAudioOffsetSPN.setValue( clip.currentAudioOffset() )
        self.uiAudioOffsetSPN.blockSignals(False)
        
        # update audio text
        self.uiAudioTXT.blockSignals(True)
        self.uiAudioTXT.setText(        clip.currentAudioSource() )
        self.uiAudioTXT.blockSignals(False)
        
        self.uiPlaybackStartSPN.setValue(   item.playbackStart() )
        self.uiPlaybackEndSPN.setValue(     item.playbackEnd() )
        
        # update source hints
        self.uiSourceStartLBL.setText(  'Min: %04i' % clip.sourceStart() )
        self.uiSourceEndLBL.setText(    'Max: %04i' % clip.sourceEnd() )
        
    def resetPlaylistOrder( self ):
        count = self.uiPlaylistTREE.topLevelItemCount()
        for i in range( count ):
            item = self.uiPlaylistTREE.topLevelItem(i)
            item.setPlaylistOrder(i)
    
    def selectAll( self ):
        self.blockUi(True)
        
        for i in range(self.uiPlaylistTREE.topLevelItemCount()):
            item = self.uiPlaylistTREE.topLevelItem(i)
            item.setSelected(True)
        
        self.blockUi(False)
        
        self.emitSelectionChanged()
    
    def selectedItems( self ):
        tree = self.uiPlaylistTREE
        
        # collect items vs. use the selectedItems to preserve the playlist ordering
        return [ tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).isSelected() ]
    
    def selectedClips( self, activeOnly = False ):
        """
                Returns the currently selected clip instances for this
                playlist.  The activeOnly flag is irrelevant in this case, as
                all clips in a playlist are considered active.  It is here
                for use with the ReviewToolWindow when collecting clips from
                different widgets.
                
                :param      activeOnly: ignored
                :type       <bool>:
                
                :return     <list> [ <Clip>, .. ]:
        """
        tree = self.uiPlaylistTREE
        
        # collect items vs. use the selectedItems to preserve the playlist ordering
        return [ tree.topLevelItem(i).clip() for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).isSelected() ]
    
    def selectedVersions( self, activeOnly = False ):
        """
                Returns the currently selected version instances for this
                playlist.  The activeOnly flag is irrelevant in this case, as
                all versions in a playlist are considered active.  It is here
                for use with the ReviewToolWindow when collecting versions from
                different widgets.
                
                :param      activeOnly: ignored
                :type       <bool>:
                
                :return     <list> [ <Version>, .. ]:
        """
        tree = self.uiPlaylistTREE
        
        # collect items vs. use the selectedItems to preserve the playlist ordering
        return [ tree.topLevelItem(i).version() for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).isSelected() ]
    
    def setContext( self, context ):
        self._context = context
        if ( context ):
            self.syncVersions(context.versions() )
            for i in range(self.uiPlaylistTREE.topLevelItemCount()):
                self.uiPlaylistTREE.topLevelItem(i).setPlaylistOrder(i)
    
    def setOptionsMenu( self, menu ):
        self.uiOptionsBTN.setMenu(menu)
    
    def setPopupMenu( self, menu ):
        self._popupMenu = menu
    
    def setSelectedVersions( self, versions ):
        self.blockUi(True)
        
        for i in range(self.uiPlaylistTREE.topLevelItemCount()):
            item = self.uiPlaylistTREE.topLevelItem(i)
            item.setSelected( item.version() in versions )
        
        self.blockUi(False)
        
        self.emitSelectionChanged()
    
    def setUseVideo( self, state = True ):
        self._useVideo = state
        self.refreshAdvanced()
    
    def showHeaderMenu( self, point ):
        # determine the column that was clicked on
        index = self.uiPlaylistTREE.header().logicalIndexAt(point)
        self._currentHeaderColumn = index
        
        # create the menu
        menu = QMenu(self)
        
        act = menu.addAction( 'Sort Ascending' )
        act.setIcon( resources.icon( 'img/playlist/sort_ascending.png' ) )
        act.triggered.connect( self.sortAscending )
        
        act = menu.addAction( 'Sort Descending' )
        act.setIcon( resources.icon( 'img/playlist/sort_descending.png' ) )
        act.triggered.connect( self.sortDescending )
        
        menu.addSeparator()
        
        # create a columns menu
        columns = menu.addMenu( 'Columns...' )
        columns.setIcon( resources.icon( 'img/playlist/columns.png' ) )
        
        for c, col in enumerate(self.COLUMNS):
            act = columns.addAction( col )
            act.setCheckable(True)
            act.setChecked( not self.uiPlaylistTREE.isColumnHidden(c) )
        
        columns.triggered.connect( self.toggleColumnTriggered )
        
        # popup the menu
        menu.exec_( QCursor.pos() )
    
    def showPopupMenu( self ):
        if ( not self._popupMenu ):
            return
        
        self._popupMenu.exec_(QCursor.pos())
    
    def sortAscending( self, column = None ):
        self.sortByColumn( column, Qt.AscendingOrder )
    
    def sortByColumn( self, column, order ):
        if ( type(column) != int ):
            column = self._currentHeaderColumn
        if ( type(column) != int ):
            return False
        
        self.uiPlaylistTREE.sortByColumn( column, order )
    
    def sortDescending( self, column = None ):
        self.sortByColumn( column, Qt.DescendingOrder )
    
    def syncVersions( self, versions ):
        # remove all the versions from the
        playlistItems       = self.playlistItems()
        version_map         = dict([(str(item.version().code()),item) for item in playlistItems if item.version() in versions])
        tree                = self.uiPlaylistTREE
        count               = len(versions)
        
        self.uiPlaylistTREE.blockSignals(True)
        
        core.updateProgress(0)
        for i, version in enumerate(versions):
            item = version_map.get(str(version.code()))
            
            # create a new icon for newly selected items
            if ( version.isActive() and not item ):
                if ( type(version) != Clip ):
                    clip = Clip(version)
                else:
                    clip = version
                    
                item = PlaylistItem(self,clip)
                item.setPlaylistOrder( clip.playlistOrder() )
                tree.addTopLevelItem(item)
                item.setSelected(True)
                
            # otherwise, take the base item
            elif ( not version.isActive() and item ):
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            
            # make sure the item is selected
            elif ( item ):
                item.setSelected(True)
            
            core.updateProgress(int((i+1)/float(count) * 100))
        
        self.blockUi(False)
        
        core.updateProgress(100)
        
        self.uiPlaylistTREE.blockSignals(False)
        
        self.setFocus()
        core.info( '%i versions synced.' % len(versions) )
    
    def switchDepartments( self, dept = '' ):
        """
                Switches the departments for the selected clips to the inputed department.
                If no department is supplied, then the user will be prompted to select one
                from a list.
                
                :param      dept:
                :type       <str>:
                
                :return     <bool>: success
        """
        # prompt the user to select a department if necessary
        if ( not dept ):
            dept, accepted = QInputDialog.getItem( self, 'Select Department', 'Select Department', settings.enabledDepartments() )
            if ( not accepted ):
                return False
        
        # make sure we have a department
        if ( not dept ):
            return False
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        self.blockUi(True)
        total = self.uiPlaylistTREE.topLevelItemCount()
        count = 0
        
        core.updateProgress(0)
        for i in range(total):
            item = self.uiPlaylistTREE.topLevelItem(i)
            
            # make sure the item is selected
            if ( not item.isSelected() ):
                core.updateProgress( int(100 * (i/float(total))) )
                continue
            
            # set the item's department
            item.setDepartment(dept)
            count += 1
            core.updateProgress( int(100 * (i/float(total))) )
        
        core.updateProgress(100)
        self.blockUi(False)
        
        QApplication.restoreOverrideCursor()
        
        self.emitPlaylistChanged()
        core.info('Switched %s clip departments.' % count)
        
        return True
    
    def toggleColumnTriggered( self, action ):
        column = str(action.text())
        hidden = not action.isChecked()
        
        # store this column as hidden for future playlists
        if ( hidden and not column in self.HIDDEN_COLUMNS ):
            self.HIDDEN_COLUMNS.append(column)
        # remove the column from the hidden list for future playlists
        elif ( not hidden and column in self.HIDDEN_COLUMNS ):
            self.HIDDEN_COLUMNS.remove(column)
        
        self.uiPlaylistTREE.setColumnHidden( self.COLUMNS.index(str(column)), not action.isChecked() )
    
    def treeWidget( self ):
        return self.uiPlaylistTREE
    
    def updateItem( self, item, column ):
        """
                Updates the item information based on the inputed column
                
                :param      item:
                :type       <QTreeWidgetItem>:
                
                :param      column:
                :type       <int>:
        """
        if ( self.uiPlaylistTREE.signalsBlocked() ):
            return
        
        self.blockUi(True)
        QApplication.setOverrideCursor( Qt.WaitCursor )
        
        # determine which column was edited
        column_name = ''
        for c, key in enumerate(self.COLUMNS):
            if ( c == column ):
                column_name = key
                break
        
        # update the department
        if ( column_name == 'Dept' ):
            if ( item.setDepartment( item.text(column) ) ):
                self.emitPlaylistChanged()
            else:
                item.setText( column, item.version().department() )
        
        # update the version
        elif ( column_name == 'Version' ):
            if ( item.setVersionName( item.text(column) ) ):
                self.emitPlaylistChanged()
            else:
                item.setText( column, item.version().name() )
        
        # update the playlist info
        elif ( column_name == 'Custom Start' ):
            item.setPlaybackStart( int(item.text(column)) )
            
        elif ( column_name == 'Custom End' ):
            item.setPlaybackEnd( int(item.text(column)) )
        
        QApplication.restoreOverrideCursor()
        self.blockUi(False)
    
    def updateToLatest( self ):
        """
                Updates the selected clips to the latest version available for their entity and department
                
                :return     <bool>: success
        """
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        self.blockUi(True)
        total = self.uiPlaylistTREE.topLevelItemCount()
        count = 0
        
        core.updateProgress(0)
        for i in range(total):
            item = self.uiPlaylistTREE.topLevelItem(i)
            
            # make sure the item is selected
            if ( not item.isSelected() ):
                core.updateProgress( int(100*(i/float(total))) )
                continue
            
            # update to latest
            item.updateToLatest()
            count += 1
        
        core.updateProgress(100)
        
        self.blockUi(False)
        QApplication.restoreOverrideCursor()
        
        core.info('Updated %s items to latest.' % count)
        
        return True
    
    def useVideo( self ):
        return self._useVideo
    
    def versions( self ):
        tree = self.uiPlaylistTREE
        return [ tree.topLevelItem(i).version() for i in range( tree.topLevelItemCount() ) ]
    
    def versionsAvailable( self ):
        return self.uiPlaylistTREE.topLevelItemCount() > 0
    
    def playlistItems( self ):
        # return all the items in the current playlist
        tree = self.uiPlaylistTREE
        return [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
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

