##
#   \namespace  reviewTool.gui.reviewtoolwindow
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/09/11
#

import datetime
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
import zipfile

import PyQt4.uic

from PyQt4.QtCore       import  QByteArray,\
                                QDir,\
                                QMimeData,\
                                QVariant,\
                                Qt,\
                                SIGNAL
                                
from PyQt4.QtGui        import  QActionGroup,\
                                QAction,\
                                QApplication,\
                                QColor,\
                                QDrag,\
                                QFileDialog,\
                                QInputDialog,\
                                QLineEdit,\
                                QMainWindow,\
                                QMenu,\
                                QMessageBox,\
                                QPixmap,\
                                QWidget

from drGadgets.gui.LoggerPanel import LoggerPanel

import rv_tools.util
import rv_tools.settings
import rv_tools

import contactSheet

# import local packages and modules

from .dialogs.filterdialog              import FilterDialog
from .dialogs.renderlookupdialog        import RenderLookupDialog
from .dialogs.rvsettingsdialog          import RVSettingsDialog
from .dialogs.shotgunplaylistdialog     import ShotgunPlaylistDialog

from .widgets.contextwidget             import ContextWidget, ContextTypeItem
from .widgets.playlistwidget            import PlaylistWidget

from ..api.contexts.playlist            import PlaylistContext
from ..api.context                      import Context
from ..api.clip                         import Clip
from ..api.entity                       import Entity
from ..api.version                      import Version
from ..api                              import contexts
from ..api.iosystem                     import IOSystem

from ..database                         import db
from ..kernel                           import core
from ..util                             import diagnosis

from ..                                 import settings
from ..                                 import resources

#--------------------------------------------------------------------------------

py_stdout = sys.stdout
py_stderr = sys.stderr

class StdOut(object):
    def write( self, msg ):
        core.infoReported.emit(msg)

class StdErr(object):
    def write( self, msg ):
        core.errorReported.emit(msg)

class CustomLoggerPanel( LoggerPanel ):
    def info( self, message="", status="", infoType = 'info' ):
        if not status: status = message
        
        message = str(message)
        status  = str(status)
        
        icon        = self.infoIcon
        color       = (200,200,200)
        textStyle   = 'black'
        
        if ( infoType == 'debug' ):
            icon        = self.debugIcon
            textStyle   = 'darkBlue'
        
        elif ( infoType == 'warn' ):
            icon        = self.warnIcon
            color       = (255,255,180)
            textStyle   = 'darkRed'
        
        elif ( infoType == 'error' ):
            color       = (255,180,180)
            textStyle   = 'darkRed'
            self.hide_progress()
            
            # remove the override cursor options
            while ( QApplication.overrideCursor() ):
                QApplication.restoreOverrideCursor()
        
        # update the icon
        self.log_icon.setPixmap(icon)
            
        # update the status information
        msg = str(status).replace('\n',' ').strip()
        if ( msg ):
            self.txtLogStatus.setText(msg)
        
        self.txtLogStatus.setCursorPosition(0)
        self.update_status_background(color)
        
        # update the detail text
        self.txtLogDetail.insertHtml('<span style="color:%s">%s</span>' % (textStyle,message.replace('\n','<br/>')) )
        self.txtLogDetail.verticalScrollBar().scroll(0, self.txtLogDetail.verticalScrollBar().maximum())
        
        # update the message count info
        self._message_count[infoType]   = self._message_count.get(infoType,0) + 1
        self._message_count['all']      = self._message_count.get('all',0) + 1

    def debug(self, message=None, status=None):
        '''
        show debug info into the logging area.
        '''
        self.info(message, status, infoType = 'debug' )
        
    def error(self, message=None, status=None):
        '''
        show error info into the logging area.
        '''
        self.info(message, status, infoType = 'error')

    def warn(self, message=None, status=None):
        '''
        show error info into the logging area.
        '''
        self.info(message, status, infoType = 'warn')

# override the standard out and error to display information to the user
# via the logging panel
#if ( not settings.DEBUG_MODE ):
#    sys.stdout = StdOut()
#    sys.stderr = StdErr()
#
#--------------------------------------------------------------------------------

class ReviewToolWindow( QMainWindow ):
    def __init__( self, parent = None ):
        super(ReviewToolWindow,self).__init__(parent)
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'ui/reviewtoolwindow.ui' )
        PyQt4.uic.loadUi( uifile, self )
        
        # force all widgets to inherit the palette properly for a mac
        if ( sys.platform == 'darwin' ):
            palette = self.palette()
            for w in self.findChildren(QWidget):
                w.setPalette(palette)
        
        # create custom properties
        self._playlistCount         = 0
        self._contextWidgetCount    = 0
        self._shotgunPlaylistDialog = ShotgunPlaylistDialog(self)
        self._diagnosisThread       = None
        self._recentFilesMenu       = None
        self._recentFiles           = []
        self._maxFiles              = 10
        
        # create the logger
        self.logger = CustomLoggerPanel(self)
        rv_tools.util.logger = self.logger
        self.uiMainSPLT.addWidget(self.logger)
        self.uiContextSPLT.setSizes( [200,1000] )
        
        # initialize the browser tree
        self.uiBrowserTREE.header().setVisible(False)
        
        # setup icons
        self.uiBrowserClearSelectionACT.setIcon(resources.icon('img/main/clear_selection.png') )
        self.uiResetACT.setIcon(                resources.icon('img/main/refresh.png') )
        self.uiFilterDepartmentsACT.setIcon(    resources.icon('img/main/filter_departments.png') )
        self.uiFilterShotsACT.setIcon(          resources.icon('img/main/filter_shots.png') )
        self.uiHelpACT.setIcon(                 resources.icon('img/main/help.png') )
        
        self.uiAddClipsACT.setIcon(             resources.icon('img/main/clips_add.png') )
        self.uiRemoveClipsACT.setIcon(          resources.icon('img/main/clips_remove.png') )
        self.uiFilterVersionsBTN.setIcon(       resources.icon('img/main/filter_versions.png') )
        self.uiOptionsBTN.setIcon(              resources.icon('img/main/options.png') )
        self.uiFilterVersionsACT.setIcon(       resources.icon('img/main/options.png') )
        self.uiEditRVSettingsACT.setIcon(       resources.icon('img/main/options.png') )
        
        self.uiContextFillACT.setIcon(          resources.icon('img/context/fill.png') )
        self.uiContextRefreshACT.setIcon(       resources.icon('img/main/refresh.png') )
        self.uiContextSelectionNewACT.setIcon(  resources.icon('img/playlist/playlist.png') )
        
        self.uiPlayACT.setIcon(                 resources.icon('img/main/play.png') )
        self.uiCompareLayoutColumnACT.setIcon(  resources.icon('img/compare/column.png') )
        self.uiCompareLayoutPackedACT.setIcon(  resources.icon('img/compare/packed.png') )
        self.uiCompareStackBlendACT.setIcon(    resources.icon('img/compare/blend.png') )
        self.uiCompareStackWipeACT.setIcon(     resources.icon('img/compare/wipe.png') )
        
        self.uiPlaylistLoadShotgunACT.setIcon(  resources.icon('img/playlist/menu/playlist_load_shotgun.png') )
        self.uiPlaylistLoadFileACT.setIcon(     resources.icon('img/playlist/menu/playlist_load_file.png') )
        self.uiPlaylistAsContextACT.setIcon(    resources.icon('img/playlist/menu/playlist_load_context.png') )
        self.uiPlaylistSaveFileACT.setIcon(     resources.icon('img/playlist/menu/playlist_save_file.png') )
        self.uiPlaylistSendToACT.setIcon(       resources.icon('img/playlist/menu/playlist_email.png') )
        self.uiPlaylistRenameACT.setIcon(       resources.icon('img/playlist/menu/playlist_rename.png') )
        self.uiReviewSessionLoadACT.setIcon(    resources.icon('img/playlist/menu/review_session_load.png') )
        self.uiReviewSessionSaveACT.setIcon(    resources.icon('img/playlist/menu/review_session_save.png') )
        self.uiAudioAddACT.setIcon(             resources.icon('img/playlist/menu/audio_add.png') )
        self.uiAudioAddByDeptACT.setIcon(       resources.icon('img/playlist/menu/audio_add.png') )
        self.uiAudioRemoveACT.setIcon(          resources.icon('img/playlist/menu/audio_remove.png') )
        self.uiSwitchDepartmentACT.setIcon(     resources.icon('img/playlist/menu/switch_department.png') )
        self.uiUpdateVersionACT.setIcon(        resources.icon('img/playlist/menu/update_version.png') )
        self.uiAddSourceFileACT.setIcon(        resources.icon('img/playlist/menu/add.png') )
        self.uiCustomRangeMENU.setIcon(         resources.icon('img/playlist/menu/custom_range.png') )
        self.uiCompareSelectionACT.setIcon(     resources.icon('img/playlist/menu/compare.png' ) )
        self.uiGenerateContactSheetACT.setIcon( resources.icon('img/playlist/menu/contact_sheet.png') )
        self.uiPlaylistSplitACT.setIcon(        resources.icon('img/playlist/menu/split.png') )
        self.uiPlaylistSelectionNewACT.setIcon( resources.icon('img/playlist/playlist.png') )
        self.uiAddTechRendersACT.setIcon(       resources.icon('img/main/options.png') )
        
        # add the profile selection
        self.uiProfileDDL.addItems( rv_tools.settings.profileNames() )
        
        # connect toolbuttons to actions
        self.uiBrowserClearSelectionBTN.setDefaultAction(   self.uiBrowserClearSelectionACT )
        self.uiResetBTN.setDefaultAction(                   self.uiResetACT )
        
        self.uiFilterDepartmentsBTN.setDefaultAction(       self.uiFilterDepartmentsACT )
        self.uiFilterShotsBTN.setDefaultAction(             self.uiFilterShotsACT )
        self.uiFilterVersionsBTN.setMenu(                   self.uiFilterVersionsMENU )
        self.uiOptionsBTN.setMenu(                          self.uiOptionsMENU )
        self.uiAddClipsBTN.setDefaultAction(                self.uiAddClipsACT )
        self.uiRemoveClipsBTN.setDefaultAction(             self.uiRemoveClipsACT )
        self.uiHelpBTN.setDefaultAction(                    self.uiHelpACT )
        self.uiPlayBTN.setDefaultAction(                    self.uiPlayACT )
        self.uiCompareBTN.setMenu(                          self.uiCompareMENU )
        self.uiCompareBTN.setDefaultAction(                 self.uiCompareStackWipeACT )
        
        # create action groups
        filterGroup = QActionGroup(self.uiFilterVersionsMENU)
        filterGroup.addAction(self.uiSortByDateACT)
        filterGroup.addAction(self.uiSortByStatusACT)
        
        compareGroup = QActionGroup(self.uiCompareMENU)
        compareGroup.addAction( self.uiCompareLayoutColumnACT )
        compareGroup.addAction( self.uiCompareLayoutPackedACT )
        compareGroup.addAction( self.uiCompareStackBlendACT )
        compareGroup.addAction( self.uiCompareStackWipeACT )
        
        self._playModeGroup = QActionGroup(self.uiOptionsMENU)
        self._playModeGroup.addAction( self.uiPlayModeCustomACT )
        self._playModeGroup.addAction( self.uiPlayModeCutACT )
        self._playModeGroup.addAction( self.uiPlayModeHandleACT )
        self._playModeGroup.addAction( self.uiPlayModeSourceACT )
        
        sourceGroup = QActionGroup(self.uiOptionsMENU)
        sourceGroup.addAction( self.uiUseMovieACT )
        sourceGroup.addAction( self.uiUseImageSequenceACT )
        
        # define the comparison methods for the actions
        self.uiCompareLayoutColumnACT.setData(  QVariant('layout') )
        self.uiCompareLayoutPackedACT.setData(  QVariant('layout_packed') )
        self.uiCompareStackBlendACT.setData(    QVariant('stack_blend') )
        self.uiCompareStackWipeACT.setData(     QVariant('stack_wipe') )
        
        # create connections
        core.infoReported.connect(      self.logger.info )
        core.warningReported.connect(   self.logger.warn )
        core.errorReported.connect(     self.logger.error )
        core.debugReported.connect(     self.logger.debug )
        core.progressUpdated.connect(   self.updateProgress )
        
        self.uiBrowserTREE.itemSelectionChanged.connect(            self.browserSelectionChanged )
        self.uiBrowserFilterDDL.lineEdit().textChanged.connect(     self.browserFilterChanged )
        self.uiBrowserFilterDDL.lineEdit().returnPressed.connect(   self.browserFilterAccepted )
        self.uiBrowserFilterDDL.activated.connect(                  self.browserFilterAccepted )
        
        self.uiContextTAB.tabCloseRequested.connect(        self.contextCloseAt )
        self.uiContextTAB.currentChanged.connect(           self.contextChanged )
        
        self.uiPlaylistTAB.currentChanged.connect(          self.playlistChanged )
        self.uiPlaylistTAB.tabCloseRequested.connect(       self.playlistCloseAt )
        
        self.uiSeparateCHK.clicked.connect(                 self.setSeparateProcess )
        
        # force this connection to use the text signal
        self.connect( self.uiProfileDDL, SIGNAL('currentIndexChanged(const QString &)'), self.setCurrentProfile )
        
        # file menu connections
        self.uiPlaylistSendToACT.triggered.connect(         self.playlistSendTo )
        self.uiExitACT.triggered.connect(                   self.close )
        
        # edit menu connections
        self.uiBrowserClearSelectionACT.triggered.connect(  self.uiBrowserTREE.clearSelection )
        self.uiResetACT.triggered.connect(                  self.clearCache )
        
        self.uiAddClipsACT.triggered.connect(               self.contextActivate )
        self.uiRemoveClipsACT.triggered.connect(            self.playlistRemoveSelection )
        
        # context menu connections
        self.uiContextFillACT.triggered.connect(            self.contextFill )
        self.uiContextRefreshACT.triggered.connect(         self.contextRefresh )
        self.uiContextSelectionNewACT.triggered.connect(    self.contextNewPlaylist )
        
        filterGroup.triggered.connect(                      self.sortingChanged )
        self.uiFilterDepartmentsACT.triggered.connect(      self.editDepartmentFilter )
        self.uiFilterShotsACT.triggered.connect(            self.editShotFilter )
        self.uiFilterVersionsACT.triggered.connect(         self.editVersionFilter )
        
        # playlist menu connections
        self._shotgunPlaylistDialog.accepted.connect(       self.playlistContextLoaded )
        self.uiPlaylistLoadShotgunACT.triggered.connect(    self.playlistLoadFromShotgun )
        self.uiPlaylistLoadFileACT.triggered.connect(       self.playlistLoadFromFile )
        self.uiPlaylistSaveFileACT.triggered.connect(       self.playlistSaveToFile )
        self.uiPlaylistRenameACT.triggered.connect(         self.playlistRename )
        
        self.uiReviewSessionLoadACT.triggered.connect(      self.sessionLoad )
        self.uiReviewSessionSaveACT.triggered.connect(      self.sessionSave )
        
        self.uiAddSourceFileACT.triggered.connect(          self.playlistAddExternalSource )
        self.uiAddTechRendersACT.triggered.connect(         self.playlistAddTechRenders )
        
        self.uiAudioAddACT.triggered.connect(               self.playlistAudioAdd )
        self.uiAudioAddByDeptACT.triggered.connect(         self.playlistAudioAddByDept )
        self.uiAudioRemoveACT.triggered.connect(            self.playlistAudioRemove )
        self.uiSwitchDepartmentACT.triggered.connect(       self.playlistSwitchDepartments )
        self.uiUpdateVersionACT.triggered.connect(          self.playlistUpdateToLatest )
        
        self.uiMatchSourceACT.triggered.connect(            self.playlistMatchSource )
        self.uiMatchCutACT.triggered.connect(               self.playlistMatchCut )
        self.uiMatchHandleACT.triggered.connect(            self.playlistMatchHandle )
        self.uiFixOutOfRangeACT.triggered.connect(          self.playlistFixOutOfRange )
        
        self.uiPlaylistAsContextACT.triggered.connect(      self.playlistNewFromContext )
        self.uiCompareSelectionACT.triggered.connect(       self.playlistCompare )
        self.uiGenerateContactSheetACT.triggered.connect(   self.playlistGenerateContactSheet )
        self.uiPlaylistSelectionNewACT.triggered.connect(   self.playlistNewSelected )
        self.uiPlaylistSplitACT.triggered.connect(          self.playlistSplit )
        
        # play menu connections
        compareGroup.triggered.connect(                     self.uiCompareBTN.setDefaultAction )
        self.uiCompareBTN.clicked.connect(                  self.playCompared )
        self.uiPlayACT.triggered.connect(                   self.play )
        
        # options menu connections
        self._playModeGroup.triggered.connect(              self.setPlayMode )
        self.uiUseMovieACT.triggered.connect(               self.updateVideoMode )
        self.uiUseImageSequenceACT.triggered.connect(       self.updateVideoMode )
        self.uiEditRVSettingsACT.triggered.connect(         self.editRVSettings )
        
        # help menu connections
        self.uiHelpACT.triggered.connect(                   self.showHelp )
        self.uiAboutACT.triggered.connect(                  self.showAbout )
        
        # initialize the data
        self.restoreSettings()
        self.refreshPlaylists()
        self.refreshRecentFiles()
        self.reset()
        
        self.uiPlaylistTAB.tabBar().installEventFilter( self )
        self.uiContextTAB.installEventFilter( self )
        
        # create the info loaded information
        core.info( 'Review Tool Loaded' )
    
    def browserContexts( self ):
        """
                Returns a list of all the contexts that are visible
                in the browser tree
                
                :return     <list> [ <Context>, .. ]:
        """
        output = []
        for i in range( self.uiBrowserTREE.topLevelItemCount() ):
            item = self.uiBrowserTREE.topLevelItem(i)
            for c in range(item.childCount()):
                output.append(item.child(c).context())
        
        return output
    
    def browserHideFiltered( self ):
        """
                Returns whether or not the filter should hide browser
                items as it applies.
                
                :return     <bool>:
        """
        return self.uiBrowserHideFilteredACT.isChecked()
    
    def browserFilters( self ):
        """
                Returns the current filters that are being used
                for the browser
                
                :return     <list> [ <str>, .. ]
        """
        ddl = self.uiBrowserFilterDDL
        return [ str(ddl.itemText(i)) for i in range(ddl.count()) ]
    
    def browserFilterAccepted( self ):
        """
                Accepts the current filtered settings for the browser
                and selects the item if necessary, logging the query
        """
        ddl = self.uiBrowserFilterDDL
        
        # make sure we have text
        entity  = ddl.lineEdit().text()
        text    = str(entity).lower().split('_')[0]
        if ( not text ):
            return False
        
        # make sure we have a valid item
        found   = None
        tree    = self.uiBrowserTREE
        for i in range( tree.topLevelItemCount() ):
            item = tree.topLevelItem(i)
            
            for c in range( item.childCount() ):
                child = item.child(c)
                if ( str(child.text(0)).lower() == text ):
                    found = child
                    break
        
        if ( not found ):
            return False
        
        # organize the list
        items   = self.browserFilters()
        search  = str(ddl.lineEdit().text())
        if ( search in items ):
            items.remove(search)
        
        # insert the new search item into the list, max out at 10 searches
        items.insert(0,search)
        self.setBrowserFilters(items)
        
        # select the item
        self.uiBrowserTREE.blockSignals(True)
        found.setSelected(True)
        self.uiBrowserTREE.blockSignals(False)
        
        self.browseTo(self.browserSelectedContexts(), entity)
    
    def browserFilterChanged( self ):
        """
                Updates the filter on the browser by changing the coloring option on
                the items in the list.
        """
        text            = str(self.uiBrowserFilterDDL.lineEdit().text()).lower().split('_')[0]
        matchColor      = QColor(255, 218, 51, 255)
        nonMatchColor   = self.palette().color( self.palette().Base )
        hidefiltered    = self.browserHideFiltered()
        first           = True
        tree            = self.uiBrowserTREE
        
        # loop through the browser tree items
        for i in range( tree.topLevelItemCount() ):
            item    = tree.topLevelItem(i)
            
            # loop through the context items
            for c in range(item.childCount()):
                child = item.child(c)
                match   = ( not text or str(child.text(0)).lower().startswith(text) )
                
                # scroll to the first found item
                if ( match and first ):
                    tree.scrollToItem(child)
                    first = False
                
                # update the color and visibility of the items
                child.setBackground( 0, matchColor if (match and text) else nonMatchColor )
                if ( hidefiltered ):
                    child.setHidden( not match )
                else:
                    child.setHidden(False)
    
    def browseTo( self, contexts, focusEntity = None ):
        """
                browses the inputed contexts within the current context
                widget.  If the focus entity is provided, then it will set focus
                and load the inputed entity by name, otherwise, the context will
                load as normal
            
                :param      contexts:
                :type       <list> [ <Context>, .. ]:
                
                :param      focusEntity:
                :type       <str> || None:
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        self.refreshEnabled()
        
        widget = self.movieContextWidget()
        widget.setVisibleContexts(contexts, focusEntity)
        
        if ( self.uiBrowserClearFilterACT.isChecked() ):
            self.uiBrowserFilterDDL.lineEdit().clear()
        
        QApplication.restoreOverrideCursor()
    
    def browserSelectionChanged( self ):
        """
                Syncs the selection between the context type tree and the current context widget
        """
        self.browseTo( self.browserSelectedContexts() )
        
    def browserSelectedContexts( self ):
        """
                Returns all the selected contexts from the browser
                
                :return     <list> [ <Context>, .. ]:
        """
        return [ item.context() for item in self.uiBrowserTREE.selectedItems() ]
    
    def browserUpdateEnabled( self ):
        """
                Updates the enabled state of the browser based on the
                currently selected context tab
        """
        self.uiBrowserFRAME.setEnabled(self.uiContextTAB.currentIndex() == 0)
    
    def clearCache( self ):
        options = QMessageBox.Yes | QMessageBox.No
        answer  = QMessageBox.question( self, 'Clear Cache', 'Refreshing will clear all cached data.  Are you sure you want to continue?', options )
        if ( answer == QMessageBox.No ):
            return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # reset the system
        self.reset(clearSelection=False)
        
        QApplication.restoreOverrideCursor()
    
    def cleanupDiagnosis( self ):
        if ( not self._diagnosisThread ):
            return
            
        # cleanup the diangosis thread
        thread = self._diagnosisThread
        self._diagnosisThread = None
        thread.terminate()
    
    def closeEvent( self, event ):
        """
                Overloads the QMainWindow closeEvent method to also
                record the current settings before shutting down.
                
                :param      event:
                :type       <QEvent:
        """
        # record the settings
        self.recordSettings()
        self.cleanupDiagnosis()
        
        # disconnect from the core to avoid seg faults
        core.infoReported.disconnect(      self.logger.info )
        core.warningReported.disconnect(   self.logger.warn )
        core.errorReported.disconnect(     self.logger.error )
        core.debugReported.disconnect(     self.logger.debug )
        core.progressUpdated.disconnect(   self.updateProgress )
        
        # close the window
        super(ReviewToolWindow,self).closeEvent( event )
    
    def contextActivate( self ):
        """
                Activates the current context selection
                
                :return     <bool>: success
        """
        contextWidget = self.currentContextWidget()
        if ( not contextWidget ):
            return False
        
        contextWidget.activateSelection()
        return True
    
    def contextChanged( self ):
        """
                Triggers the events that need to happen when the context
                widget changes.
        """
        self.browserUpdateEnabled()
        self.contextSync()
    
    def contextClose( self ):
        """
                Closes the current context view for the tool
        """
        self.contextCloseAt(self.uiContextTAB.currentIndex())
    
    def contextCloseAt( self, index ):
        """
                Closes the context widget at the given index
                
                :return     <bool>
        """
        
        if ( self.uiContextTAB.tabText(index) == 'Movie Context' ):
            QMessageBox.information( self, 'Cannot Close Movie Context', 'You cannot close the movie context for the Review Tool.' )
            return False
        
        curr_index  = self.uiContextTAB.currentIndex()
        count       = self.uiContextTAB.count()
        
        # calculate the next index
        if ( curr_index < index ):
            new_index = curr_index
        elif ( index < curr_index ):
            new_index = curr_index - 1
        elif ( index == count - 2 ):
            new_index = index - 1
        else:
            new_index = index
            
        # update the playlist
        self.uiContextTAB.blockSignals(True)
        self.uiContextTAB.removeTab(index)
        self.uiContextTAB.setCurrentIndex(new_index)
        self.uiContextTAB.tabBar().setVisible(self.uiContextTAB.count() > 1)
        self.uiContextTAB.blockSignals(False)
        
        self.contextChanged()
        
        return True
    
    def contextFill( self ):
        contextWidget = self.currentContextWidget()
        if ( contextWidget ):
            dept, accepted = QInputDialog.getItem( self, 'Select Department', 'Select Department', settings.enabledDepartments() )
            if ( not accepted ):
                return False
            
            contextWidget.activateBetweenExtents(dept)
    
    def contextNew( self, name = '' ):
        """
                Creates a new context view for the tool
                
                :param      name:   tab name for the new context widget
                :type       str:
                
                :return     <ContextWidget>:
        """
        # create the context widget
        widget = ContextWidget(self)
        
        # force all widgets to inherit the palette properly for a mac
        if ( sys.platform == 'darwin' ):
            palette = self.palette()
            for w in [widget] + widget.findChildren(QWidget):
                w.setPalette(palette)
        
        widget.setMenu( self.uiContextMENU )
        
        # create connections
        widget.activeStateChanged.connect(          self.playlistSync )
        widget.versionSelectionChanged.connect(     self.playlistSyncSelection )
        widget.versionDoubleClicked.connect(        self.playVersions )
        widget.loadingStateChanged.connect(         self.contextSyncWithoutSelection )
        
        # add the tab
        self._contextWidgetCount += 1
        if ( not name ):
            name = 'Context %02i' % self._contextWidgetCount
        
        # insert the tab at the end
        self.uiContextTAB.blockSignals(True)
        self.uiContextTAB.addTab( widget, name )
        self.uiContextTAB.setCurrentIndex(self.uiContextTAB.count() - 1)
        self.uiContextTAB.tabBar().setVisible(self.uiContextTAB.count() > 1 )
        self.uiContextTAB.blockSignals(False)
        
        self.browserUpdateEnabled()
        
        return widget
    
    def contextNewPlaylist( self ):
        contextWidget = self.currentContextWidget()
        if ( not contextWidget ):
            return
            
        versions = contextWidget.selectedVersions()
        
        # create the new playlist
        self.uiPlaylistTAB.blockSignals(True)
        self.uiPlaylistTAB.setUpdatesEnabled(False)
        
        contextWidget.blockSignals(True)
        
        playlist = self.playlistNew()
        playlist.blockSignals(True)
        
        contextWidget.setActiveVersions(versions)
        playlist.syncVersions(versions)
        
        contextWidget.blockSignals(False)
        playlist.blockSignals(False)
        
        self.uiPlaylistTAB.blockSignals(False)
        self.uiPlaylistTAB.setUpdatesEnabled(True)
    
    def contextRefresh( self ):
        contextWidget = self.currentContextWidget()
        if ( contextWidget ):
            contextWidget.refreshSelection()
    
    def contextSync( self, saveSelection = False ):
        """
                Syncs the versions from the current playlist to the current context widget
                
                :return     <bool>: success
        """
        playlist        = self.currentPlaylist()
        contextWidget   = self.currentContextWidget()
        
        # make sure we have a playlist and context widget
        if ( not (playlist and contextWidget) ):
            return False
        
        contextWidget.blockSignals(True)
        contextWidget.setActiveVersions(    playlist.versions() )
        if ( not saveSelection ):
            contextWidget.setSelectedVersions(  playlist.selectedVersions() )
        contextWidget.blockSignals(False)
        
        self.refreshEnabled()
        return True
    
    def contextSyncWithoutSelection( self ):
        self.contextSync(saveSelection=True)
    
    def contextSyncSelection( self ):
        """
                Syncs the current context selection from the current playlist's selection
                
                :return     <bool>: success
        """
        playlist        = self.currentPlaylist()
        contextWidget   = self.currentContextWidget()
        
        if ( not (playlist and contextWidget) ):
            self.refreshEnabled()
            return False
        
        contextWidget.blockSignals(True)
        contextWidget.setSelectedVersions( playlist.selectedVersions() )
        contextWidget.blockSignals(False)
        
        self.refreshEnabled()
        return True
    
    def currentContextWidget( self ):
        """
                Returns the current context tree widget that is active in the tabs
                
                :return <ContextWidget> || None
        """
        return self.uiContextTAB.currentWidget()
    
    def currentClips( self ):
        """
                Collects a list of the currently active clips based on the selection
                and focus for the different widgets.
                
                :return     <list> [ <Clip>, .. ]
        """
        # collect the current context and playlist widgets
        contextWidget   = self.currentContextWidget()
        playlist        = self.currentPlaylist()
        clips           = []
        
        # attempt to use the focused widget
        if ( playlist and contextWidget and not contextWidget.hasFocus() ):
            clips = playlist.selectedClips()
        
        # otherwise, use the contextWidget
        if ( not clips and contextWidget ):
            clips = [ Clip(version) for version in contextWidget.selectedVersions() ]
        
        # return the current clips
        return clips
        
    def currentVersions( self ):
        """
                Collects a list of the currently active clips based on the selection
                and focus for the different widgets.
                
                :return     <list> [ <Clip>, .. ]
        """
        # collect the current context and playlist widgets
        contextWidget   = self.currentContextWidget()
        playlist        = self.currentPlaylist()
        clips           = []
        
        # attempt to use the focused widget
        if ( playlist and contextWidget and not contextWidget.hasFocus() ):
            clips = playlist.selectedVersions()
        
        # otherwise, use the contextWidget
        if ( not clips and contextWidget ):
            clips = contextWidget.selectedVersions()
        
        # return the current clips
        return clips
        
    def currentPlaylist( self ):
        """
                Returns the currently selected playlist widget from the
                playlist tab
                
                :return <PlaylistWidget> || None:
        """
        widget = self.uiPlaylistTAB.currentWidget()
        if ( type(widget) == PlaylistWidget ):
            return widget
        return None
    
    def editDepartmentFilter( self ):
        # edit the department filters
        if ( FilterDialog.edit( settings.departmentFilters(), parent = self ) ):
            # reload the columns that are visible for the context widgets
            for i in range( self.uiContextTAB.count() ):
                context = self.uiContextTAB.widget(i)
                context.refreshColumns()
    
    def editRVSettings( self ):
        # edit the RV Settings
        if ( RVSettingsDialog.edit( parent = self ) ):
            self.setCurrentProfile(     rv_tools.settings.value('current_profile') )
            self.setSeparateProcess(    rv_tools.settings.value('separate_process') )
            
    def editShotFilter( self ):
        if ( FilterDialog.edit( settings.shotFilters(), parent = self, sortable = False ) ):
            self.reset(clearSelection=False)
    
    def editVersionFilter( self ):
        if ( FilterDialog.edit( settings.versionFilters(), parent = self ) ):
            self.sortingChanged()
    
    def eventFilter( self, object, event ):
        text = self.uiPlaylistTAB.tabText(self.uiPlaylistTAB.currentIndex())
        
        # start the drag event
        if ( object == self.uiPlaylistTAB.tabBar() and event.type() == event.MouseMove ):
            # store the playlist as a pixmap
            pixmap = QPixmap( self.uiPlaylistTAB.size() )
            self.uiPlaylistTAB.render( pixmap )
            
            # create the mime data
            data = QMimeData()
            data.setText( text )
            
            # create the drag event
            drag = QDrag( self )
            drag.setPixmap( pixmap )
            drag.setMimeData( data )
            drag.exec_()
        
        # accept the drag enter event
        elif ( object == self.uiContextTAB and event.type() == event.DragEnter ):
            if ( text == event.mimeData().text() ):
                event.accept()
                return True
        
        # accept the drop event
        elif ( object == self.uiContextTAB and event.type() == event.Drop ):
            if ( text == event.mimeData().text() ):
                playlist = self.currentPlaylist()
                self.playlistNewFromContext(playlist.context())
                event.accept()
                return True
        
        return False
    
    def focusedWidget( self ):
        contextWidget   = self.currentContextWidget()
        playlist        = self.currentPlaylist()
        
        # attempt to use the focused widget
        if ( playlist and contextWidget and not contextWidget.hasFocus() ):
            return playlist
            
        return contextWidget
        
    def gotoNextContextWidget( self ):
        """
                Sets the current index for the context view to the next one in the list,
                wrapping around to the front if it is at the end
        """
        index = self.uiContextTAB.currentIndex() + 1
        if ( index == self.uiContextTAB.count() ):
            index = 0
        self.uiContextTAB.setCurrentIndex(index)
    
    def gotoPrevContextWidget( self ):
        """
                Sets the current index for the context view to the prev one in the list,
                wrapping to the end if it is at the end
        """
        index = self.uiContextTAB.currentIndex() - 1
        if ( index < 0 ):
            index = self.uiContextTAB.count() - 1
        self.uiContextTAB.setCurrentIndex(index)
    
    def initializeFromCLI( self, options, clips ):
        """
                Initializes the interface based on the inputed command line
                arguments
                
                :param      options:
                :type       <OptionParserDict> || None:
                
                :param      clips:
                :type       <list> [ <str> ]:
        """
        # make sure we have some options
        if ( not options ):
            return False
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        
        # load the playlist if it exists
        if ( options.playlist ):
            self.playlistLoadFromFile( options.playlist )
        elif ( clips ):
            clips    = Clip.fromData(   clips, 
                                        defaultDepartment   = options.dept,
                                        padLeft             = options.padLeft,
                                        padRight            = options.padRight,
                                        overrideAudio       = options.audio )
            
            # add the clips to the playlist
            for i, clip in enumerate(clips):
                clip.setPlaylistOrder(i)
            
            # grab the current playlist
            playlist = self.currentPlaylist()
            playlist.syncVersions( clips )
            self.refreshEnabled()
        
        QApplication.restoreOverrideCursor()
    
    def movieContextWidget( self ):
        return self.uiContextTAB.widget(0)
    
    def navigateTo( self, code ):
        """
                Looks up the version item in the movie context based on the
                inputed code
                
                :param  code:
                :type   <str>:
        """
        # make sure we're in the movie context
        if ( self.uiContextTAB.currentIndex() ):
            return None
        
        # grab the scene info
        found = False
        scene = code.split('_')[0]
            
        # navigate to the proper scene
        found   = False
        browser = self.uiBrowserTREE
        for i in range( browser.topLevelItemCount() ):
            item = browser.topLevelItem(i)
            for c in range( item.childCount() ):
                child = item.child(c)
                if ( child.text(0) == scene ):
                    child.setSelected(True)
                    found = True
                    break
                    
            if ( found ):
                break
        
        # navigate to the selected version
        if ( found ):
            item = self.currentContextWidget().navigateTo( code )
            if ( item ):
                self.contextSyncSelection()
                return item
        return None
    
    def play( self ):
        return self.playClips( self.currentClips() )
    
    def playMode( self ):
        return str(self._playModeGroup.checkedAction().property('mode').toString())
    
    def playCompared( self ):
        # determine the default action to use when comparing
        self.playClips( self.currentClips(), self.playCompareMethod() )
    
    def playCompareMethod( self ):
        return str( self.uiCompareBTN.defaultAction().data().toString() )
    
    def playClips( self, clips, compareMethod = None ):
        """
                Playst the inputed clips with the current RV settings and protocols.
                If the compareMethod is supplied, then the RV session will be laucnhed
                in a comparison way, otherwise, it will be launched in a standard way.
                
                :param      clips:
                :type       <list> [ <Clip>, .. ]:
                
                :param      compareMethod:
                :type       <str>:
                
                :return     <bool>: success
        """
        if ( not clips ):
            return False
            
        QApplication.setOverrideCursor( Qt.WaitCursor )
        
        # generate the playlist sequence
        self.logger.reset()
        
        Clip.playClips( clips, compareMethod, createSession = self.uiSeparateCHK.isChecked() )
        
        core.info('Play successful.')
        QApplication.restoreOverrideCursor()
    
    def playVersions( self, versions ):
        self.playClips( [ Clip(version) for version in versions ] )
    
    def playlistAddExternalSource( self ):
        """
                Prompts the user to select a new image sequence or file to add to the
                current review playlist directly.
                
                :return     <bool>: success
        """
        # make sure we have a playlist
        playlist = self.currentPlaylist()
        if ( not playlist ):
            return False
        
        filter      = settings.videoFileTypes()
        filenames   = QFileDialog.getOpenFileNames( self, 'Import Source Sequence - Select Frame Source or Movie', QDir.currentPath(), filter )
        
        if ( not filenames ):
            return False
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.logger.reset()
        
        context     = playlist.context()
        clips       = []
        
        for filename in filenames:
            clip = Clip(Version(Entity(None,'','',0,''), {}))
            clip.setActive(True)
            clip.setCustomVideo( filename )
            clip.setVideoMode(     Clip.VideoMode.Custom )
            clip.setPlaybackStart( clip.sourceStart() )
            clip.setPlaybackEnd(   clip.sourceEnd() )
            clips.append(clip)
            
        # sync the latest versions to the playlist
        playlist.syncVersions(clips)
        
        self.refreshEnabled()
        
        QApplication.restoreOverrideCursor()
        
        return True
    
    def playlistAddTechRenders( self ):
        """
            \remarks    prompts the user to select corresponding technical renders
                        based on the selected renders
        """
        playlist    = self.currentPlaylist()
        versions    = RenderLookupDialog.collectVersions( playlist.selectedVersions(), self )
        if ( versions ):
            playlist.syncVersions(versions)
    
    def playlistAudioAdd( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.overrideAudio(True)
    
    def playlistAudioAddByDept( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            dept, accepted = QInputDialog.getItem( self, 'Select Department', 'Select Department', settings.enabledDepartments() )
            if ( not accepted ):
                return False
            
            playlist.overrideAudio(True,dept = dept)
    
    def playlistAudioRemove( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.overrideAudio(False)
    
    def playlistChanged( self ):
        """
                Hanldes when a user changes the current playlist by creating a new playlist
                and/or syncing the context's active and selected versions based on the
                current playlist.
        """
        # check if the user clicked on the 'add' tab
        if ( self.uiPlaylistTAB.currentIndex() == self.uiPlaylistTAB.count() - 1 ):
            self.playlistNew()
        
        # sync the current context
        self.contextSync()
    
    def playlistClose( self ):
        """
                Closes the current playlist, provided that the current widget
                in the playlist tab is a PlaylistWidget type
        """
        return self.playlistCloseAt(self.uiPlaylistTAB.currentIndex())
    
    def playlistCloseAt( self, index ):
        """
                Closes the playlist at the given index
                
                :param  index:
                :type   int:
                
                :return     bool:
        """
        widget      = self.uiPlaylistTAB.widget(index)
        curr_index  = self.uiPlaylistTAB.currentIndex()
        count       = self.uiPlaylistTAB.count()
        
        # make sure its a playlist widget
        if ( type(widget) != PlaylistWidget ):
            return False
        
        # make sure we will have at least 1 left over
        elif ( count == 2 ):
            QMessageBox.information( self, 'Cannot Close', 'You need to always have at least 1 playlist avaiable.' )
            return False
        
        # calculate the next index
        if ( curr_index < index ):
            new_index = curr_index
        elif ( index < curr_index ):
            new_index = curr_index - 1
        elif ( index == count - 2 ):
            new_index = index - 1
        else:
            new_index = index
            
        # update the playlist
        self.uiPlaylistTAB.blockSignals(True)
        self.uiPlaylistTAB.removeTab(index)
        self.uiPlaylistTAB.setCurrentIndex(new_index)
        self.uiPlaylistTAB.blockSignals(False)
        
        self.contextSync()
        
        return True
    
    def playlistCompare( self ):
        """
            \remarks    compares the data from all the different playlists
                        using the breakdown diagnosis package
        """
        QApplication.setOverrideCursor( Qt.WaitCursor )
        
        # create a collection of data for the breakdown diagnosis to process
        collection      = {}
        total_entities  = 0
        
        # loop through the playlists
        for i in range( self.uiPlaylistTAB.count() - 1 ):
            playlist    = self.uiPlaylistTAB.widget(i)
            tab         = str(self.uiPlaylistTAB.tabText(i))
            data        = []
            
            # loop through the clips
            for clip in playlist.clips():
                sinfofile   = clip.sinfoFilepath()
                
                # retrieve the sinfo file from the clip
                if ( not sinfofile ):
                    continue
                
                title = '%s   ver: %s   updated_at: %s' % (clip.displayName(),clip.name(),clip.createdAt())
                data.append( (title,sinfofile) )
            
            # add the data and the tab to the file
            if ( data ):
                collection[tab] = data
                total_entities += len(data)
        
        # generate a rough time estimate for the user
        time_estimate   = '%d min %d sec' % (total_entities * 17 // 60, total_entities * 40 % 60)
        
        # generate the question to present to the user
        question        = []
        question.append( 'You are going to generate the breakdown-playlist version comparison document.' )
        question.append( 'It will be put in a folder on your desktop:' )
        question.append( '~/Desktop/breakdown_diagnosis' )
        question.append( '' )
        question.append( '    Total shots to be processed: %d.' % total_entities )
        question.append( '    Time estimation: %s.' % time_estimate )
        question.append( '' )
        question.append( 'Do you wish to continue?' )
        
        QApplication.restoreOverrideCursor()
        
        options         = QMessageBox.Yes | QMessageBox.No
        answer          = QMessageBox.question( self, 'Breakdown Version Comparison', '\n'.join(question), options )
        
        if ( answer == QMessageBox.No ):
            return
        
        # kill the last thread if it exists
        self.cleanupDiagnosis()
        
        # otherwise, process the results
        self._diagnosisThread = diagnosis.DiagnosisThread(collection)
        self._diagnosisThread.finished.connect( self.showDiagnosisFinished )
        self._diagnosisThread.terminated.connect( self.cleanupDiagnosis )
        self._diagnosisThread.start()
    
    def playlistContextLoaded( self, context ):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        playlist = self.currentPlaylist()
        if ( not playlist.isEmpty() ):
            playlist = self.playlistNew(context.name())
        else:
            self.playlistRename(context.name())
            
        playlist.setContext(context)
        
        if ( context.shotgunId() ):
            self.uiPlaylistTAB.setTabIcon( self.uiPlaylistTAB.currentIndex(), resources.icon('img/playlist/menu/playlist_load_shotgun.png') )
        
        self.contextSync()
        QApplication.restoreOverrideCursor()
    
    def playlistGenerateContactSheet( self ):
        self.logger.reset()
        system = IOSystem.findByType('.rv')
        if ( not system ):
            QMessageBox.critical( self, 'Missing Exporter', 'Missing the .rv file exporter.' )
            return False
        
        QApplication.setOverrideCursor( Qt.WaitCursor )

        self.logger.debug('Importing contactSheet module...')

        # make sure to set the context for the contact sheet to be the review tool
        from contactSheet.generator             import ContactSheetGenerator
        from contactSheet.gui.generatorwindow   import ContactSheetGeneratorWindow

        self.logger.debug('Restoring previous settings...')

        # restore the last generator based on user preferences
        generator = ContactSheetGenerator.restoreLast('Review Tool')

        self.logger.debug('Saving out playlist file...')

        # save out a temp RV file to use
        filename = settings.tempPath('rv_playlist.rv')
        playlist = self.currentPlaylist()
        
        system.save( filename, playlist.clips() )
        
        self.logger.debug('Loading playlist for Contact Sheet')
        # set it as the generator's playlist
        generator.setPlaylist(filename)

        # edit the generator
        ContactSheetGeneratorWindow.edit(generator, parent = self)
        
        self.logger.info('Loaded playlist contact sheet...')
        
        QApplication.restoreOverrideCursor()
    
    def playlistLoadRecent( self, action ):
        """
                Loads the filename from the recent list
                
                :param      action:
                :type       <QAction>:
        """
        self.playlistLoadFromFile( str(action.data().toString()) )
    
    def playlistLoadFromFile( self, filename = '' ):
        """
                Prompts the user to load a new playlist from a given XML or RV file.
                
                :return     <bool>
        """
        if ( not (filename and type(filename) == str) ):
            filetypes   = 'Review Tool Files (*%s)' % ' *'.join( [ system.fileType() for system in IOSystem.systems() if system.imports() ] )
            filename    = str(QFileDialog.getOpenFileName( self, 'Import Playlist', QDir.currentPath(), filetypes ))
            
        if ( not (filename and os.path.isfile(filename)) ):
            return False
            
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.logger.reset()
        
        context = PlaylistContext.fromFile( filename )
        if ( not context ):
            QApplication.restoreOverrideCursor()
            return False
        
        # create a playlist context from file
        QDir.setCurrent( os.path.dirname(str(filename)) )
        
        # record recent file
        self.recordRecentFile( filename )
        
        # load the playlist in the ui
        self.playlistContextLoaded(context)
        QApplication.restoreOverrideCursor()
        
        ecount = self.logger.message_count('error')
        wcount = self.logger.message_count('warn')
        if ( ecount or wcount ):
            self.logger.warn('%s warnings and %s errors occurred.' % (wcount,ecount))
        
        return True
    
    def playlistLoadFromShotgun( self ):
        self._shotgunPlaylistDialog.show()
        self._shotgunPlaylistDialog.refresh()
    
    def playlistMatchSource( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.matchCustomRanges( playlist.selectedItems(), 'source' )
    
    def playlistMatchCut( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.matchCustomRanges( playlist.selectedItems(), 'edit' )
    
    def playlistMatchHandle( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.matchCustomRanges( playlist.selectedItems(), 'handle' )
    
    def playlistFixOutOfRange( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.fixCustomRanges( playlist.selectedItems() )
    
    def playlistOptionsMenu( self ):
        """
                Creates a new playlist menu to be assigned to the playlist menu
                button for a new playlist widget.  This will take various actions
                defined within the main ReviewToolWindow and assign them to a
                menu all together.  This is mostly for backwards compatibility.
                
                :return     <QMenu>:
        """
        menu = QMenu(self)
        
        # add playlist specific file actions
        menu.addAction( self.uiPlaylistLoadShotgunACT )
        menu.addAction( self.uiPlaylistLoadFileACT )
        menu.addAction( self.uiPlaylistSaveFileACT )
        menu.addAction( self.uiPlaylistSendToACT )
        menu.addAction( self.uiPlaylistRenameACT )
        menu.addSeparator()
        menu.addAction( self.uiReviewSessionLoadACT )
        menu.addAction( self.uiReviewSessionSaveACT )
        menu.addSeparator()
        
        # add all of the playlist menu actions
        for action in self.uiPlaylistMENU.actions():
            menu.addAction(action)
        
        return menu
        
    def playlistPopupMenu( self ):
        menu = QMenu(self)
        
        # add all of the playlist menu actions
        for action in self.uiPlaylistMENU.actions():
            menu.addAction(action)
        
        return menu
    
    def playlistNew( self, name = '' ):
        """
                Handle the playlist change event triggered when the user
                clicks on a particular playlist tab
        """
        # create the playlist widget
        widget = PlaylistWidget(self)
        
        # force all widgets to inherit the palette properly for a mac
        if ( sys.platform == 'darwin' ):
            palette = self.palette()
            for w in [widget] + widget.findChildren(QWidget):
                w.setPalette(palette)
        
        # set widget options
        widget.setPopupMenu( self.playlistPopupMenu() )
        widget.setOptionsMenu( self.playlistOptionsMenu() )
        widget.setUseVideo( self.useVideo() )
        
        # create connections
        widget.selectionChanged.connect(        self.contextSyncSelection )
        widget.playlistChanged.connect(         self.contextSync )
        widget.versionLookupRequested.connect(  self.navigateTo )
        
        # add the tab
        self._playlistCount += 1
        if ( not name ):
            name = 'Playlist %02i' % self._playlistCount
        
        # determine the index to add to
        index = self.uiPlaylistTAB.count() - 1
        self.uiPlaylistTAB.blockSignals(True)
        self.uiPlaylistTAB.insertTab( index, widget, name )
        self.uiPlaylistTAB.setTabIcon( index, resources.icon('img/playlist/playlist.png') )
        self.uiPlaylistTAB.setCurrentIndex( index )
        self.uiPlaylistTAB.blockSignals(False)
        
        return widget
    
    def playlistNewSelected( self ):
        """
                Create a new playlist based on the current selection
                
                :return <PlaylistWidget>
        """
        playlist = self.currentPlaylist()
        if ( not playlist ):
            return
            
        versions = playlist.selectedVersions()
        playlist = self.playlistNew()
        playlist.syncVersions(versions)
        return playlist
    
    def playlistSendTo( self ):
        """
                Saves the current playlist to a temp file and then loads the
                rv_tools.sendplaylistdialog class to send the playlist to another
                person
        """
        import rv_tools.sendmessage
        
        filename = settings.sharePath( 'rv_playlist_share_%i.rv' % time.mktime( datetime.datetime.now().timetuple() ) )
        if ( not self.playlistSaveToFile( filename ) ):
            QMessageBox.critical( self, 'Could not save Playlist', 'Could not save playlist to file %s' % filename )
            return False
        
        return rv_tools.sendmessage.sendPlaylist( filename, parent = self )
    
    def playlistSplit( self ):
        """
                Create new playlists from the current selection based on
                their various sections
        """
        playlist = self.currentPlaylist()
        if ( not playlist ):
            return
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        
        # split the vesions up into sections
        section_order   = []
        sections        = {}
        for clip in playlist.selectedVersions():
            name        = clip.entity().name()
            section     = name.split('_')[0]
            
            # create a new section
            if ( not section in section_order ):
                section_order.append(section)
                sections[section] = []
            
            sections[section].append(clip)
        
        index = self.uiPlaylistTAB.currentIndex()
        self.uiPlaylistTAB.setUpdatesEnabled(False)
        self.uiPlaylistTAB.blockSignals(True)
        
        # create the new playlist sections
        for section in section_order:
            playlist = self.playlistNew(section)
            playlist.syncVersions(sections[section])
            
        self.uiPlaylistTAB.setCurrentIndex(index)
        self.uiPlaylistTAB.setUpdatesEnabled(True)
        self.uiPlaylistTAB.blockSignals(False)
        
        QApplication.restoreOverrideCursor()
    
    def playlistNewFromContext( self, context ):
        """
                Creates a new context view for the current playlist's context
                
                :return     <ContextWidget>
        """
        if ( not context ):
            QMessageBox.information( self, 'Cannot Load Context', 'You can only load playlist contexts for playlists loaded from file or Shotgun.' )
            return None
            
        playlist    = self.currentPlaylist()
        context     = playlist.context()
        
        contextWidget = self.contextNew( name = context.name() )
        contextWidget.addContext(context)
        self.contextSync()
        
        return contextWidget
    
    def playlistRename( self, new_name = '' ):
        """
                Prompts the user to rename the current playlist.blockSignals
                
                :return     <bool>: success
        """
        playlist = self.currentPlaylist()
        
        # make sure we're not renaming a shotgun playlist
        if ( playlist.context() and playlist.context().shotgunId() ):
            QMessageBox.information( self, 'Shotgun Playlist', 'Shotgun playlist cannot be renamed.' )
            return False
        
        curr_tab                = self.uiPlaylistTAB.currentIndex()
        curr_name               = self.uiPlaylistTAB.tabText( curr_tab )
        
        if ( type(new_name) != str ):
            new_name, accepted      = QInputDialog.getText( self, 'Playlist Name', 'Enter name for playlist:', QLineEdit.Normal, curr_name )
        else:
            accepted                = True
        
        if ( accepted and new_name ):
            self.uiPlaylistTAB.setTabText( curr_tab, new_name )
            return True
        return False
    
    def playlistRemoveSelection( self ):
        """
                Removes the currently selected versions from the current playlist
                
                return      <bool>:
        """
        # make sure we have a valid playlist
        playlist        = self.currentPlaylist()
        if ( not playlist ):
            return False
        
        # remove the selected versions from the list
        playlist.removeVersions(playlist.selectedVersions())
        
        # sync the current context widget
        contextWidget   = self.currentContextWidget()
        if ( contextWidget ):
            self.contextSync(saveSelection = True)
            contextWidget.setFocus()
        
        # update the enabled options
        self.refreshEnabled()
        return True
        
    def playlistSaveToFile( self, filename = '' ):
        """
                Prompts the user to save the current playlist out to a given XML or RV file.
                
                :return     <bool>: success
        """
        if ( not (type(filename) in (str,unicode) and filename) ):
            filetypes = ';;'.join( [ '%s (*%s)' % (system.systemName(),system.fileType()) for system in IOSystem.systems() if system.exports() ] )
            filename = str(QFileDialog.getSaveFileName( self, 'Export Playlist', QDir.currentPath(), filetypes ))
            
        if ( not filename ):
            return False
        
        # make sure to get an extension
        ftype = os.path.splitext(filename)[1]
        if ( not ftype ):
            filename += '.rtool'
            ftype = '.rtool'
        
        # determine the proper system to save with
        system = IOSystem.findByType(ftype)
        if ( system ):
            return system.save( filename, self.currentPlaylist().clips() )
        return False
    
    def playlistSwitchDepartments( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.switchDepartments()
    
    def playlistSync( self, versions = None ):
        """
                Syncs the current versions from the context widget (or provided
                versionsWidget) to the current playlist.
                
                :param      versionsWidget:
                :type       <ContextVersionsWidget>:
        """
        # updates the current playlist from the current versions widget
        playlist = self.currentPlaylist()
        if ( not playlist ):
            QMessageBox.critical( self, 'Missing Playlist', 'No playlist was found to work with.' )
            return False
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        #self.logger.reset()
        
        # sync the current versions from the versions widget
        if ( versions == None ):
            contextWidget = self.currentContextWidget()
            
            if ( not contextWidget ):
                versions = []
            else:
                versions = contextWidget.selectedVersions(activeOnly=True)
        
        # sync the playlist versions
        playlist.syncVersions(versions)
        self.playlistSyncSelection()
        
        QApplication.restoreOverrideCursor()
    
    def playlistSyncSelection( self ):
        """
                Syncs the current playlist selection to the current context widget's
                selection
                
                :return     <bool>: success
        """
        playlist        = self.currentPlaylist()
        contextWidget   = self.currentContextWidget()
        
        if ( not (playlist and contextWidget) ):
            self.refreshEnabled()
            return False
        
        playlist.blockSignals(True)
        playlist.setSelectedVersions( contextWidget.selectedVersions( activeOnly = True ) )
        playlist.blockSignals(False)
        
        self.refreshEnabled()
        
        return True
    
    def playlistUpdateToLatest( self ):
        playlist = self.currentPlaylist()
        if ( playlist ):
            playlist.updateToLatest()
    
    def recordRecentFile( self, filename ):
        """
                Records the filename as a recently opened file and then
                refreshes the list of files
                
                :param      filename:
                :type       <str>:
        """
        filename = str(filename)
        if ( filename in self._recentFiles ):
            self._recentFiles.remove(filename)
        
        self._recentFiles.insert(0,filename)
        self._recentFiles = self._recentFiles[:self._maxFiles]
        self.refreshRecentFiles()
    
    def recordSettings( self ):
        # record the rv_tools settings changes
        rv_tools.settings.save()
        
        # record the review tool settings
        options = {}
        options['GUI::current_path']    = QDir.currentPath()
        options['GUI::geometry']        = self.saveGeometry().toBase64()
        options['GUI::context_sizes']   = ','.join( [str(size) for size in self.uiContextSPLT.sizes()] )
        options['GUI::main_sizes']      = ','.join( [str(size) for size in self.uiMainSPLT.sizes()] )
        options['GUI::browser_filters'] = ','.join( self.browserFilters() )
        options['GUI::hide_filter']     = self.browserHideFiltered()
        options['GUI::sort_by_date']    = self.sortByDate()
        options['GUI::play_mode']       = self.playMode()
        options['GUI::compare_method']  = self.playCompareMethod()
        options['GUI::use_movie']       = self.useVideo()
        options['GUI::playlist_hidden_cols']    = ':'.join(PlaylistWidget.HIDDEN_COLUMNS)
        options['GUI::recent_files']            = ':'.join(self._recentFiles)
        options['GUI::clear_filter']            = self.uiBrowserClearFilterACT.isChecked()
        
        # save the review tool settings
        return settings.save(options)
    
    def refreshBrowser( self ):
        """
                Refreshes the context types that are loaded in the main tree
        """
        
        # load the context choices
        self.uiBrowserTREE.setUpdatesEnabled(False)
        self.uiBrowserTREE.blockSignals(True)
        
        self.uiBrowserTREE.clear()
        
        for name in Context.contextTypeNames():
            # create the context type item
            item = ContextTypeItem( name, Context.contextType(name) )
            
            # add the context to the tree
            self.uiBrowserTREE.addTopLevelItem( item )
            item.loadContexts()
            
        self.uiBrowserTREE.blockSignals(False)
        self.uiBrowserTREE.setUpdatesEnabled(True)
    
    def refreshContextWidgets( self ):
        """
                Initializes the context widgets that will be used for this review tool session
        """
        self._contextWidgetCount = 0
        
        self.uiContextTAB.blockSignals(True)
        self.uiContextTAB.clear()
        
        # define the main movie context
        widget = self.contextNew('Movie Context')
        widget.setContexts( self.browserContexts() )
        
        self.uiContextTAB.blockSignals(False)
        
    def refreshEnabled( self ):
        # collect the widgets
        contextWidget   = self.currentContextWidget()
        playlist        = self.currentPlaylist()
        focused         = self.focusedWidget()
        
        # collect the available states
        focused_set     = False if focused == None          else focused.versionsAvailable()
        context_set     = False if contextWidget == None    else contextWidget.versionsAvailable()
        playlist_set    = False if playlist == None         else playlist.versionsAvailable()
        
        # update the play options based on the focus set
        self.uiPlayACT.setEnabled(          focused_set )
        self.uiContextFillACT.setEnabled(   focused_set )
        self.uiContextSelectionNewACT.setEnabled( focused_set )
        self.uiCompareMENU.setEnabled(      focused_set )
        self.uiCompareBTN.setEnabled(       focused_set )
        
        # update the clip options based on the context set
        self.uiAddClipsACT.setEnabled(      context_set )
        self.uiAddClipsACT.setToolTip( 'Add selected context clips to playlist' if context_set else 'Waiting for selected clips to load...' )
        
        # update the action options based on the playlist set
        self.uiRemoveClipsACT.setEnabled(       playlist_set )
        self.uiRemoveClipsACT.setToolTip( 'Remove selected clips from playlist' if playlist_set else '' )
        
        self.uiAddTechRendersACT.setEnabled(    playlist_set )
        self.uiSwitchDepartmentACT.setEnabled(  playlist_set )
        self.uiUpdateVersionACT.setEnabled(     playlist_set )
        self.uiAudioAddACT.setEnabled(          playlist_set )
        self.uiAudioRemoveACT.setEnabled(       playlist_set )
    
    def refreshPlaylists( self ):
        """
                Initialize the playlist widgets that will be used for this review tool session
        """
        self._playlistCount = 0
        
        self.uiPlaylistTAB.blockSignals(True)
        self.uiPlaylistTAB.clear()
        self.uiPlaylistTAB.addTab(QWidget(self),'New')
        self.playlistNew()
        self.uiPlaylistTAB.blockSignals(False)
    
    def refreshRecentFiles( self ):
        """
                Refreshes the recent files menu with the latest files
        """
        if ( not self._recentFiles ):
            return
            
        if ( not self._recentFilesMenu ):
            self.uiFileMENU.addSeparator()
            self._recentFilesMenu = self.uiFileMENU.addMenu( 'Recent files' )
            self._recentFilesMenu.triggered.connect( self.playlistLoadRecent )
        
        # clear the current actions
        self._recentFilesMenu.clear()
        
        # add the file actions
        for i, filename in enumerate( self._recentFiles ):
            act = self._recentFilesMenu.addAction( '%i: %s' % (i+1,os.path.basename( filename )) )
            act.setData( QVariant(filename) )
    
    def reset( self, clearSelection = True ):
        """
                Performs a full reset that will wipe all settings and reload the
                interface fresh
        """
        
        # record current browser selection
        tree        = self.uiBrowserTREE
        selItems    = [ str(item.text(0)) for item in tree.selectedItems() ]
        
        # clear the cache info
        db.clearCache()
        
        # refresh the data
        self.refreshBrowser()
        self.refreshContextWidgets()
        self.refreshEnabled()
        
        if ( clearSelection ):
            return True
        
        # restore the browser selection
        tree.blockSignals(True)
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            for c in range(item.childCount()):
                child = item.child(c)
                if ( child.text(0) in selItems ):
                    child.setSelected(True)
        tree.blockSignals(False)
        self.browserSelectionChanged()
        
    
    def restoreSettings( self ):
        # restore the rv_tools settings
        rv_tools.settings.restore()
        
        self.setCurrentProfile(     rv_tools.settings.value('current_profile') )
        self.setSeparateProcess(    rv_tools.settings.value('separate_process') )
        
        # restore the review tool settings
        options = settings.restore()
        
        # restore geometry options
        if ( options.has_key('GUI::geometry') ):
            self.restoreGeometry( QByteArray.fromBase64( options['GUI::geometry'] ) )
        if ( options.has_key('GUI::context_sizes') ):
            self.uiContextSPLT.setSizes( [ int(size) for size in options['GUI::context_sizes'].split(',') ] )
        if ( options.has_key('GUI::main_sizes') ):
            self.uiMainSPLT.setSizes( [ int(size) for size in options['GUI::main_sizes'].split(',') ] )
        
        # restore playlist columns
        PlaylistWidget.HIDDEN_COLUMNS = [ col for col in options.get('GUI::playlist_hidden_cols','').split(':') if col ]
        
        # restore preference options
        QDir.setCurrent(            options.get('GUI::current_path',QDir.currentPath()) )
        self.setRecentFiles(        options.get('GUI::recent_files','').split(':') )
        self.setBrowserFilters(     options.get('GUI::browser_filters','').split(',') )
        self.setBrowserHideFiltered(options.get('GUI::hide_filter') != 'False' )
        self.setSortByDate(         options.get('GUI::sort_by_date') != 'False' )
        self.setPlayMode(           options.get('GUI::play_mode') )
        self.setPlayCompareMethod(  options.get('GUI::compare_method') )
        self.setUseVideo(           options.get('GUI::use_movie') != 'False' )
        self.uiBrowserClearFilterACT.setChecked( options.get('GUI::clear_filter') != 'False' )
        
        return True
    
    def sessionLoad( self ):
        """
                Prompts the user to select a filepath location to restore a zipped
                review session from.
                
                :return     <bool>: success
        """
        filename = str(QFileDialog.getOpenFileName( self, 'Load Review Session', QDir.currentPath(), 'RV Session Files (*.zip)' ))
        if ( not filename ):
            return False
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        self.logger.reset()
        
        core.info('Loading session from: %s' % filename)
        
        # make sure we have a proper extension
        if ( not filename.endswith('.zip') ):
            filename += '.zip'
            
        self.uiPlaylistTAB.blockSignals(True)
        self.uiPlaylistTAB.setUpdatesEnabled(False)
        
        # clear the current playlists
        for i in range(self.uiPlaylistTAB.count() - 2, -1, -1):
            self.uiPlaylistTAB.removeTab(i)
        
        # create the temp directory if it doesn't already exist
        temp_dir = settings.tempPath( os.path.basename(filename).split('.')[0] )
        if ( not os.path.exists(temp_dir) ):
            os.mkdir(temp_dir)
        
        # extract the zipped data
        zipper  = zipfile.ZipFile(filename,'r')
        zipdata = [ (basename,zipper.read(basename)) for basename in zipper.namelist() ]
        zipper.close()
        
        # sort alphanumerically
        zipdata.sort( lambda x,y: cmp( x[0], y[0] ) )
        
        # restore the files
        for basename, content in zipdata:
            # extract the id information from the base name
            results = re.match( '(?P<index>\d+)(_shotgunId(?P<shotgun_id>\d+))?_(?P<tab_name>.*)', basename.split('.')[0] )
            if ( not results ):
                core.warn('Improper session name: %s' % basename)
                continue
                
            # create a temp rv file
            playlist_file   = os.path.join( temp_dir, basename )
            options         = results.groupdict()
            tab_name        = options['tab_name']
            shotgun_id      = options['shotgun_id']
            
            # save the contents out to a temp file
            f = open(playlist_file,'w')
            f.write(content)
            f.close()
            
            # load the playlist from the file
            if ( not self.playlistLoadFromFile( playlist_file ) ):
                core.warn('Error loading playlist from file: %s' % playlist_file)
                continue
            
            self.playlistRename( new_name = tab_name )
            
            # if the playlist is a shotgun one, then update the information about it
            if ( shotgun_id ):
                self.currentPlaylist().context().setShotgunId( int(shotgun_id) )
        
        # remove the unzipped paths
        shutil.rmtree(temp_dir)
        
        self.uiPlaylistTAB.blockSignals(False)
        self.uiPlaylistTAB.setUpdatesEnabled(True)
        
        QApplication.restoreOverrideCursor()
        return True
        
    def sessionSave( self ):
        """
                Prompts the user to select a filepath location to save a zipped 
                version of this review session to.
                
                :return     <bool>: success
        """
        filename = str(QFileDialog.getSaveFileName( self, 'Save Review Session', QDir.currentPath(), 'RV Session Files (*.zip)' ))
        if ( not filename ):
            return False
        
        system = IOSystem.findByType('.rtool')
        if ( not system ):
            QMessageBox.critical( self, 'Missing IO System', 'Could not find the .rvtool exporter.')
            return False
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        self.logger.reset()
        
        # make sure we have a proper extension
        if ( not filename.endswith('.zip') ):
            filename += '.zip'
        
        # create the temp directory
        temp_dir = os.path.join( settings.TEMP_PATH, os.path.basename(filename).split('.')[0] )
        if ( not os.path.exists(temp_dir) ):
            os.mkdir(temp_dir)
        
        # loop through each playlist and save it to the session folder
        zipdata = []
        for index in range( self.uiPlaylistTAB.count() - 1 ):
            playlist        = self.uiPlaylistTAB.widget(index)
            context         = PlaylistWidget.context()
            shotgun_id      = '_shotgunId%s' % context.shotgunId() if context and context.shotgunId() else ''
            tab_name        = str(self.uiPlaylistTAB.tabText(index))
            
            playlist_name   = '%s%s_%s.rtool' % (index,shotgun_id,tab_name)
            playlist_path   = os.path.join(temp_dir,playlist_name)
            
            core.info('Saving %s...' % playlist_path)
            if ( system.save( playlist_path, playlist.clips() ) ):
                zipdata.append((playlist_path,playlist_name))
            else:
                core.warn('Could not save file for: %s' % playlist_tab)
        
        # save the zip data
        zipper = zipfile.ZipFile(filename,'w')
        for filepath, basename in zipdata:
            zipper.write( filepath, basename )
        zipper.close()
        
        # clean out the temp file
        shutil.rmtree( temp_dir )
        
        QApplication.restoreOverrideCursor()
        return True
    
    def setBrowserFilters( self, filters, maxCount = 10 ):
        """
                Sets the items that have been used as filters
                for the current browser filter combo box
                
                :param  filters:
                :type   <list> [ <str>, .. ]:
        """
        ddl = self.uiBrowserFilterDDL
        
        # update the items in the list
        ddl.blockSignals(True)
        ddl.lineEdit().blockSignals(True)
        ddl.clear()
        ddl.addItems([ filter for filter in list(set(filters))[:maxCount] if filter ])
        ddl.lineEdit().blockSignals(False)
        ddl.blockSignals(False)
        
        # clear the current edit
        ddl.lineEdit().clear()
    
    def setBrowserHideFiltered( self, state ):
        """
                Sets the hide filtered state for the browser
                to the inputed state
                
                :param      state:
                :type       <bool>:
        """
        self.uiBrowserHideFilteredACT.setChecked(state)
        self.browserFilterChanged()
    
    def setCurrentProfile( self, text ):
        """
                Sets the current profile in both the ui and
                the settings
                
                :param      text:
                :type       <str>:
        """
        # update the settings profile
        rv_tools.settings.setValue( 'current_profile', text )
        
        # update the ui profile
        self.uiProfileDDL.blockSignals(True)
        self.uiProfileDDL.setCurrentIndex( self.uiProfileDDL.findText( text ) )
        self.uiProfileDDL.blockSignals(False)
    
    def setPlayCompareMethod( self, method ):
        """
                Sets the current compare action on the compare
                button to the inputed method.
                
                :param      method:
                :type       <str>:
                
                :return     <bool>: success
        """
        for action in self.uiCompareMENU.actions():
            if ( action.data().toString() == method ):
                self.uiCompareBTN.setDefaultAction(action)
                return True
        return False
    
    def setPlayMode( self, mode ):
        # extract the mode from the inputed action
        if ( type(mode) == QAction ):
            mode = mode.property('mode').toString()
            
        settings.PLAY_MODE = str(mode)
        
        for act in self._playModeGroup.actions():
            if ( act.property('mode').toString() == mode ):
                act.setChecked(True)
                return
        
        # set to the custom playmode
        settings.PLAY_MODE = str(self.uiPlayModeCustomACT.property('mode').toString())
        self.uiPlayModeCustomACT.setChecked(True)
    
    def setRecentFiles( self, filenames ):
        """
                Sets the recent file list for this window to the inputd list of
                files
        """
        filenames = [str(filename) for filename in filenames if filename]
        filenames = filenames[:self._maxFiles]
        self._recentFiles = filenames
        self.refreshRecentFiles()
    
    def setSortByDate( self, state ):
        """
                Sets whether or not the sort by date option should be enabled
                based on the inputed state.
                
                :param      state:
                :type       <bool>:
        """
        self.uiSortByDateACT.setChecked(state)
        self.uiSortByStatusACT.setChecked(not state)
        
        settings.SORT_VERSIONS_BY_DATE = state
        
        self.sortingChanged()
    
    def setSeparateProcess( self, state ):
        """
                Sets the current process state in both the ui and
                the settings
                
                :param      settings:
                :type       <bool>:
        """
        # update the settings process
        rv_tools.settings.setValue( 'separate_process', state )
        
        # update the ui process
        self.uiSeparateCHK.blockSignals(True)
        self.uiSeparateCHK.setChecked(state)
        self.uiSeparateCHK.blockSignals(False)
    
    def setUseVideo( self, state = True ):
        """
            Changes the use video state for the review tool
            
            :param      state:
            :type       <bool>:
        """
        settings.MEDIA_MODE = 'video' if state else 'image'
        
        # update the state for the use movie action
        self.uiUseMovieACT.setChecked(state)
        self.uiUseImageSequenceACT.setChecked(not state)
        
        # update the playlist widgets
        for i in range(self.uiPlaylistTAB.count()-1):
            widget = self.uiPlaylistTAB.widget(i)
            if ( not isinstance(widget,PlaylistWidget) ):
                continue
                
            widget.setUseVideo(state)
    
    def showAbout( self ):
        """
                Launches the about dialog
        """
        pass
    
    def showDiagnosisFinished( self ):
        QMessageBox.information( self, 'Diagnosis Analysis Finished', 'Your Package/Breakdown version comparison has completed and can be found at ~/Desktop/breakdown_diagnosis' )
        self.cleanupDiagnosis()
    
    def showHelp( self ):
        """
                Launches the help window in a new browser

        """
        webbrowser.open( settings.PATH_HELPDOCS )
    
    def sortByDate( self ):
        """
            Returns whether or not the tool should sort review
            items by their date.
            
            :return     <bool>:
        """
        return self.uiSortByDateACT.isChecked()
    
    def sortingChanged( self ):
        """
                Resets the sorting order for versions within each context widget
        """
        settings.SORT_VERSIONS_BY_DATE = self.uiSortByDateACT.isChecked()
        
        for i in range( self.uiContextTAB.count() ):
            widget = self.uiContextTAB.widget(i)
            
            if ( type(widget) == QWidget ):
                continue
            
            widget.refreshVersionSorting()
    
    def updateProgress( self, percent ):
        if ( percent == 100 ):
            self.logger.hide_progress()
        else:
            self.logger.show_progress( 100, percent )
    
    def updateVideoMode( self ):
        self.setUseVideo( self.useVideo() )
    
    def useVideo( self ):
        return self.uiUseMovieACT.isChecked()
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

