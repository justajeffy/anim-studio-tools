##
#   \namespace  reviewTool.gui.widgets.contextwidget.contextentityitem
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

from PyQt4.QtCore   import  QSize, Qt, QTimer
from PyQt4.QtGui    import  QTreeWidgetItem

from ....                       import settings
from ....database               import db, threads
from ....api.version            import Version
from ..loaderwidget             import LoaderWidget
from .contextversionswidget     import ContextVersionsWidget

class ContextEntityItem( QTreeWidgetItem ):
    def __init__( self, entity ):
        super(ContextEntityItem,self).__init__( [entity.name()] )
        
        self.setSizeHint( 0, QSize( 0, 24 ) )
            
        self._entity            = entity
        self._versionsThread    = threads.EntityVersionsThread(db.project(),Version.ShotgunFields,entity.tankKey())
        self._loaded            = False
        self._loading           = False
        self._lazyTimer         = QTimer()
        self._lazyTimer.setInterval(1000)
        
        self._lazyTimer.timeout.connect( self.loadVersions )
        self._versionsThread.finished.connect( self.updateData )
    
    def activeVersions( self ):
        tree = self.treeWidget()
        if ( not tree ):
            return []
            
        output = []
        
        for i in range(1,tree.columnCount()):
            widget = tree.itemWidget(self,i)
            if ( not widget ):
                continue
                
            output += widget.activeVersions()
        
        return output
    
    def entity( self ):
        return self._entity
    
    def contextWidget( self ):
        return self.treeWidget()
    
    def initData( self ):
        if ( not self.readyToLoad() ):
            self.lazyLoadVersions()
        else:
            self.loadVersions()
         
    def isLoading( self ):
        return self._loading
    
    def reload( self ):
        self._loaded = False
        self.loadVersions( True )
       
    def lazyLoadVersions( self ):
        self._lazyTimer.start()
    
    def loadVersions( self, force = False ):
        # make sure the item should be loaded
        # (by default forcing, so it will load in the
        # background vs. when the user displays)
        if ( not (force or self.readyToLoad()) ):
            return False
            
        if ( self._loaded ):
            self._lazyTimer.stop()
            return
        
        self._loaded    = False
        tree            = self.treeWidget()
        
        # make sure the lazy timer isn't going anymore
        self._lazyTimer.stop()
        
        for c in range( 1, tree.columnCount() ):
            tree.setItemWidget( self, c, LoaderWidget( tree ) )
        
        # start the thread
        self._loading   = True
        self.contextWidget().markLoading(self,True)
        self._versionsThread.start()
            
    def readyToLoad( self ):
        tree = self.treeWidget()
        if ( not tree ):
            return False
        
        geom    = tree.geometry()
        rect    = tree.visualItemRect(self)
        visible = (geom.top() <= rect.bottom() and rect.top() <= geom.bottom())
        
        # check to see the visible state of the item
        if ( not (visible or self.isSelected()) ):
            return False
        
        return True
    
    def reSort( self ):
        tree    = self.treeWidget()
        
        entity  = self.entity()
        entity.sortVersions()
        
        for i in range( tree.columnCount() ):
            widget = tree.itemWidget(self,i)
            
            if ( isinstance( widget, ContextVersionsWidget ) ):
                widget.setVersions( entity.findVersions(tree.departmentAt(i)) )
    
    def updateData( self ):
        # mark that the item has been loaded
        self._lazyTimer.stop()
        self._loaded    = True
        self._loading   = False
        
        tree = self.treeWidget()
        
        # set the data for the entity
        self._entity.setVersionsFromShotgun( self._versionsThread.versions() )
        
        for c in range( 1, tree.columnCount() ):
            versions = self._entity.findVersions(tree.departmentAt(c))
            if ( versions ):
                tree.setItemWidget( self, c, ContextVersionsWidget( tree, self._entity, versions ) )
            else:
                tree.setItemWidget( self, c, None )
        
        # unmark this item as loading
        self.contextWidget().markLoading(self,False)
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

