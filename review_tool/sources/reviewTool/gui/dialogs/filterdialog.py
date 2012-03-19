##
#   \namespace  reviewTool.gui.dialogs.filterdialog
#
#   \remarks    Defines a dialog class for editing the department selection and order for a context widget
#   
#   \author     Dr. D Studios
#   \date       08/03/11
#

import os.path
import types

import PyQt4.uic

from PyQt4.QtCore   import  Qt, QSize
from PyQt4.QtGui    import  QDialog,\
                            QTreeWidgetItem

from ...            import resources

class FilterDialog( QDialog ):
    def __init__( self, parent = None ):
        super(FilterDialog,self).__init__(parent)
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'ui/filterdialog.ui' )
        PyQt4.uic.loadUi( uifile, self )
        self.setPalette(parent.palette())
        
        # initialize the tree
        header = self.uiFilterTREE.header()
        header.setResizeMode( 0, header.Stretch )
        header.setResizeMode( 1, header.ResizeToContents )
        
        # safe-bind the dropEvent method for the tree widget (not handled by eventFilter in Qt)
        bound_method = types.MethodType( self.handleDropEvent.im_func, self.uiFilterTREE, self.uiFilterTREE.__class__ )
        self.uiFilterTREE.__dict__[ 'dropEvent' ] = bound_method
        
        # create custom properties
        self._checkable     = True
        self._sortable      = True
        self._filterOptions = []
    
    def accept( self ):
        # record the current order and enabled properties for the current filters
        for i in range( self.uiFilterTREE.topLevelItemCount() ):
            item    = self.uiFilterTREE.topLevelItem(i)
            option  = self._filterOptions.get(str(item.text(1)))
            if ( not option ):
                continue
            
            option['enabled']   = item.checkState(0) == Qt.Checked
            option['order']     = i
        
        super(FilterDialog,self).accept()
    
    def filterOptions( self ):
        return self._filterOptions
    
    def handleDropEvent( tree, event ):
        source  = tree.currentItem()
        target  = tree.itemAt(event.pos())
        
        # make sure we have a valid move
        if ( source and target and source != target ):
            # take the source out of the tree
            tree.takeTopLevelItem( tree.indexOfTopLevelItem( source ) )
            
            # insert the item back into the tree
            tree.insertTopLevelItem( tree.indexOfTopLevelItem( target ), source )
            
            # reset the current item as the dropped item
            tree.setCurrentItem(source)
    
    def isCheckable( self ):
        return self._checkable
    
    def isSortable( self ):
        return self._sortable
    
    def refresh( self ):
        tree = self.uiFilterTREE
        
        tree.blockSignals(True)
        tree.setUpdatesEnabled(False)
        
        tree.clear()
        
        # create the code color
        codeClr = tree.palette().color(tree.palette().Base).darker(140)
        
        # sort the data hash
        filters = self._filterOptions.items()
        filters.sort( lambda x,y: cmp(x[1].get('order',100000),y[1].get('order',100000)) )
        
        for key, option in filters:
            # create the data for this item
            data = { 'code': key }
            data.update(option)
            
            # create the data item
            item = QTreeWidgetItem( [data.get('name',''),data.get('code','')] )
            item.setIcon( 0, resources.icon( data.get('icon','') % data ) )
            item.setSizeHint( 0, QSize( 0, 18 ) )
            
            # check the item if its enabled
            if ( self.isCheckable() ):
                enabled = data.get('enabled',False)
                item.setCheckState( 0, Qt.Checked if enabled else Qt.Unchecked )
            
            # set some ui data for the code name to hide it a bit
            item.setTextAlignment( 1, Qt.AlignRight | Qt.AlignCenter )
            item.setForeground( 1, codeClr )
            
            # add to the tree
            tree.addTopLevelItem(item)
        
        tree.setUpdatesEnabled(True)
        tree.blockSignals(False)
    
    def setCheckable( self, state = True ):
        self._checkable = state
        
    def setFilterOptions( self, options ):
        self._filterOptions = options
    
    def setSortable( self, state = True ):
        self._sortable = state
        
        if ( state ):
            self.uiFilterTREE.setDragDropMode( self.uiFilterTREE.DragDrop )
        else:
            self.uiFilterTREE.setDragDropMode( self.uiFilterTREE.NoDragDrop )
    
    # define static methods
    @staticmethod
    def edit( options, sortable = True, checkable = True, parent = None ):
        # create a dialog to edit the departments
        dlg = FilterDialog( parent )
        
        dlg.setFilterOptions(options)
        dlg.setSortable(sortable)
        dlg.setCheckable(checkable)
        
        dlg.refresh()
        
        if ( dlg.exec_() ):
            return True
        return False
        
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

