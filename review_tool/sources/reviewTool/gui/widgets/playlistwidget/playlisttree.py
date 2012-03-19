##
#   \namespace  reviewTool.gui.widgets.playlistwidget.playlisttree
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

from PyQt4.QtGui    import  QItemDelegate,\
                            QTreeWidget

class PlaylistTree( QTreeWidget ):
    def __init__( self, *args ):
        super(PlaylistTree,self).__init__( *args )
        
        # make sure the last column isn't stretching
        header = self.header()
        header.setStretchLastSection(False)
        
        # set custom properties
        self._editableColumns = []
        
    def closeEditor( self, editor, hint ):
        """
                Overloads the base QTreeWidget closeEditor method
                to handle tabbing properly when editing.
                
                :param      editor:
                :type       <QWidget>:
                
                :param      hint:
                :type       <QItemDelegate>:
        """
        # grab the current index
        index       = self.currentIndex()
        
        # look for the next editable column
        row         = index.row()
        column      = index.column()
        
        model       = self.model()
        num_rows    = model.rowCount()
        num_columns = model.columnCount()
        
        # edit the next item by moving left >> right in the row vs. up >> down in the column
        if ( hint == QItemDelegate.EditNextItem ):
            colStart    = 0
            colEnd      = num_columns
            colDelta    = 1
            rowDelta    = 1
        
        # edit the previous item by moving right >> left in the row vs. down >> up in the column    
        elif ( hint == QItemDelegate.EditPreviousItem ):
            colStart    = num_columns - 1
            colEnd      = -1
            colDelta    = -1
            rowDelta    = -1
            
        else:
            super(PlaylistTree,self).closeEditor( editor, hint )
            return
        
        # close the editor with no hint
        super(PlaylistTree,self).closeEditor( editor, QItemDelegate.NoHint )
        
        # loop through the rows and columns looking for the next editable item
        colCurrent  = column + colDelta
        found       = False
        
        while ( not found and (0 <= row and row < num_rows) ):
            # loop through the columns
            for c in range( colCurrent, colEnd, colDelta ):
                if ( self.isColumnEditable( c ) ):
                    column  = c
                    found   = True
                    break
            
            # increment the row and reset the column
            if ( not found ):
                row         += rowDelta
                colCurrent  = colStart
        
        # make sure we have another index
        if ( not found ):
            return
        
        # make sure we have a valid index
        next = index.sibling( row, column )
        if ( not (next and next.isValid()) ):
            return
        
        # edit the next index found
        self.setCurrentIndex(next)
        super(PlaylistTree,self).edit(next)
    
    def edit( self, index, trigger, event ):
        """
                Overloads the base edit item method to disable editing of 
                items based on this widget's editable column list.
                
                :param      index:
                :type       <QModelIndex>:
                
                :param      trigger:
                :type       <QTreeWidget.EditTrigger>
                
                :param      event:
                :type       <QEvent>
                
                :return     <bool>: started edit
        """
        if ( self.isColumnEditable( index.column() ) ):
            return super(PlaylistTree,self).edit(index,trigger,event)
        return False
    
    def isColumnEditable( self, column ):
        """
                Returns whether or not the inputed column is editable
                
                :param      column:
                :type       <int>:
                
                :return     <bool>:
        """
        return column in self._editableColumns
        
    def setEditableColumns( self, columns ):
        """
                Sets the columns that are going to be editable to the
                inputed list of columns
                
                :param      columns:
                :type       <list> [ <int>, ]]:
        """
        self._editableColumns = columns
        
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

