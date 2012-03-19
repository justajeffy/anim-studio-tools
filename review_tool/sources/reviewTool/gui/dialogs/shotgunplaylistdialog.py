##
#   \namespace  reviewTool.gui.dialogs.shotgunplaylistdialog
#
#   \remarks    Loads the playlist from shotgun as a new playlist in the Review Tool
#   
#   \author     Dr. D Studios
#   \date       08/05/11
#

import datetime
import os.path

import PyQt4.uic
from PyQt4.QtCore               import  Qt,\
                                        QDate,\
                                        QThread,\
                                        QSize,\
                                        pyqtSignal
                                        
from PyQt4.QtGui                import  QApplication,\
                                        QDialog,\
                                        QTreeWidgetItem

from ..widgets.loaderwidget     import LoaderWidget
from ..delegates.griddelegate   import GridDelegate
from ...api.contexts.playlist   import PlaylistContext
from ...database                import db, Database
from ...                        import settings
from ...                        import resources

#--------------------------------------------------------------------------------

class PlaylistLookupThread( QThread ):
    def __init__( self ):
        super(PlaylistLookupThread,self).__init__()
        
        self.department     = ''
        self.filterDate     = False
        self.filterClosed   = True
        self.date           = None
        self.cache          = {}
    
    def run( self ):
        # create a new database connection
        dbthread        = Database()
        
        # lookup the department
        dept            = dbthread.findDepartment(self.department)
        
        # create the shotgun lookup
        fields      = ['id','sg_department','code','sg_date_and_time','description']
        filters     = [['sg_department','is', dept ]]
        
        if ( self.filterClosed ):
            filters.append( ['sg_status','is_not','clsd'] )
        
        if ( self.filterDate ):
            filters.append( ['sg_date_and_time', 'greater_than', datetime.datetime.fromordinal(self.date.toordinal()) ] )
        
        # create a new session
        sg_playlists                    = dbthread.session().find( 'Playlist', filters, fields )
        self.cache[self.department]     = sg_playlists
        
#--------------------------------------------------------------------------------

class PlaylistItem( QTreeWidgetItem ):
    def __init__( self, sg_playlist ):
        super(QTreeWidgetItem,self).__init__()
        
        self._sg_playlist = sg_playlist
        
        self.setSizeHint( 0, QSize( 0, 18 ) )
        self.setText(0,sg_playlist['sg_department']['name'])
        self.setText(1,sg_playlist['code'])
        
        created = sg_playlist['sg_date_and_time']
        if ( created ):
            self.setText(2,created.strftime('%y-%m-%d %H:%M:%S'))
        else:
            self.setText(2,'.....................')
        
        comments = sg_playlist['description']
        if ( comments ):
            self.setText(3, comments)
        else:
            self.setText(3,'.....................')
        
    def createPlaylistContext( self ):
        QApplication.setOverrideCursor( Qt.WaitCursor )
        # loads the playlist context for the shotgun data
        fields          = [ 'versions' ]
        sg_playlist     = db.session().find_one( 'Playlist', [['id','is',self._sg_playlist['id']]], ['versions'] )
        sg_playlist.update( self._sg_playlist )
        QApplication.restoreOverrideCursor()
        
        return PlaylistContext( sg_playlist['code'], sg_playlist )
    
    def filterBy( self, text ):
        if ( not text ):
            self.setHidden(False)
            return
            
        for i in range(self.columnCount()):
            if ( text in str(self.text(i)).lower() ):
                self.setHidden(False)
                
                # select the first item when filtering
                if ( len(self.treeWidget().selectedItems()) == 0 ):
                    self.treeWidget().setCurrentItem(self)
                    
                return
        
        self.setHidden(True)

#--------------------------------------------------------------------------------

class ShotgunPlaylistDialog( QDialog ):
    accepted = pyqtSignal(object)
    
    def __init__( self, parent = None ):
        super(ShotgunPlaylistDialog,self).__init__(parent)
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'ui/shotgunplaylistdialog.ui' )
        PyQt4.uic.loadUi( uifile, self )
        
        # set the palette and icons
        if ( parent ):
            self.setPalette(parent.palette())
        
        self.uiRefreshBTN.setIcon( resources.icon('img/main/refresh.png') )
        
        # default to the last 2 weeks
        self.uiPlaylistDATE.setDate( QDate.currentDate().addDays( -14 ) )
        
        # initialize the tree
        self.uiPlaylistTREE.sortByColumn(2,Qt.DescendingOrder)
        self.uiPlaylistTREE.setItemDelegate( GridDelegate( self.uiPlaylistTREE ) )
        header = self.uiPlaylistTREE.header()
        for i in range(self.uiPlaylistTREE.columnCount()-1):
            header.setResizeMode(i,header.ResizeToContents)
        
        # set the department information
        self._departments = settings.departments()
        self._departments.sort()
        self.uiDepartmentDDL.addItems( settings.departmentLabels(self._departments) )
        
        # create the lookup thread
        self._lookupThread = PlaylistLookupThread()
        self._lookupThread.finished.connect( self.refreshPlaylists )
        
        # create connections
        self.uiOkBTN.clicked.connect(                       self.accept )
        self.uiCancelBTN.clicked.connect(                   self.reject )
        self.uiDepartmentDDL.currentIndexChanged.connect(   self.refresh )
        self.uiRefreshBTN.clicked.connect(                  self.clearCache )
        self.uiPlaylistTREE.itemDoubleClicked.connect(      self.accept )
        self.uiPlaylistTREE.itemSelectionChanged.connect(   self.refreshEnabled )
        self.uiDateCHK.toggled.connect(                     self.clearCache )
        self.uiFilterClosedCHK.clicked.connect(             self.clearCache )
        self.uiPlaylistDATE.dateChanged.connect(            self.clearCache )
        self.uiFilterTXT.textChanged.connect(               self.filterResults )
    
    def accept( self ):
        super(ShotgunPlaylistDialog,self).accept()
        
        self._lookupThread.terminate()
        items = self.uiPlaylistTREE.selectedItems()
        self.accepted.emit(self.currentPlaylist())
    
    def clearCache( self ):
        self._lookupThread.cache.clear()
        self.refresh()
    
    def closeEvent( self, event ):
        self._lookupThread.terminate()
        super(ShotgunPlaylistDialog,self).closeEvent(event)
    
    def currentDepartment( self ):
        return self._departments[self.uiDepartmentDDL.currentIndex()]
    
    def currentPlaylist( self ):
        item = self.uiPlaylistTREE.currentItem()
        if ( item ):
            return item.createPlaylistContext()
        return None
    
    def filterResults( self, text ):
        self.uiPlaylistTREE.blockSignals(True)
        self.uiPlaylistTREE.setUpdatesEnabled(False)
        self.uiPlaylistTREE.clearSelection()
        text = str(text).lower()
        for i in range( self.uiPlaylistTREE.topLevelItemCount() ):
            self.uiPlaylistTREE.topLevelItem(i).filterBy(text)
        
        self.uiPlaylistTREE.blockSignals(False)
        self.uiPlaylistTREE.setUpdatesEnabled(True)
        
        self.refreshEnabled()
    
    def refresh( self ):
        # terminate the current thread
        self._lookupThread.blockSignals(True)
        self._lookupThread.terminate()
        self._lookupThread.blockSignals(False)
        
        dept = self.currentDepartment()
        
        if ( not dept in self._lookupThread.cache ):
            # set the loader widget overlay
            LoaderWidget.start(self.uiPlaylistTREE, 'Loading %s Playlists from Shotgun...' % self.uiDepartmentDDL.currentText())
            
            # start the lookup
            self._lookupThread.department   = self.currentDepartment()
            self._lookupThread.filterDate   = self.uiDateCHK.isChecked()
            self._lookupThread.filterClosed = self.uiFilterClosedCHK.isChecked()
            self._lookupThread.date         = self.uiPlaylistDATE.date().toPyDate()
            self._lookupThread.start()
            
            self.uiOkBTN.setEnabled(False)
        else:
            self.refreshPlaylists()
    
    def refreshPlaylists( self ):
        # make sure we have a valid department
        dept = self.currentDepartment()
        if ( not dept in self._lookupThread.cache ):
            return
        
        # stop the loader widget overlay
        LoaderWidget.stop(self.uiPlaylistTREE)
        
        self.uiPlaylistTREE.blockSignals(True)
        self.uiPlaylistTREE.setUpdatesEnabled(False)
        
        self.uiPlaylistTREE.clear()
        for sg_playlist in self._lookupThread.cache[dept]:
            self.uiPlaylistTREE.addTopLevelItem(PlaylistItem(sg_playlist))
        
        self.filterResults( self.uiFilterTXT.text() )
        self.uiPlaylistTREE.setUpdatesEnabled(True)
        self.uiPlaylistTREE.blockSignals(False)
    
    def refreshEnabled( self ):
        self.uiOkBTN.setEnabled( len(self.uiPlaylistTREE.selectedItems()) == 1 )
        
    def reject( self ):
        self._lookupThread.terminate()
        super(ShotgunPlaylistDialog,self).reject()
    
    def show( self ):
        super(ShotgunPlaylistDialog,self).show()
        
        self.uiFilterTXT.setFocus()
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

