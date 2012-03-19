##
#   \namespace  reviewTool.gui.widgets.playlistwidget.delegates.deptdelegate
#
#   \remarks    Defines the DeptDelegate class used when rendering department information
#               in the playlistwidget
#   
#   \author     Dr. D Studios
#   \date       08/09/11
#

from PyQt4.QtCore               import  QVariant,\
                                        Qt
                                        
from PyQt4.QtGui                import  QComboBox

from ....delegates.griddelegate import GridDelegate
from .....                      import settings

class DeptDelegate(GridDelegate):
    def __init__( self, parent ):
        super(DeptDelegate,self).__init__( parent )
    
    def createEditor( self, parent, option, index ):
        """
                Creates the QComboBox widget to use when editing department
                selections
                
                :param      parent:
                :type       <QWidget>:
                
                :param      option:
                :type       <QStyleOptionViewItem>:
                
                :param      index:
                :type       <QModelIndex>:
        """
        if ( not index.isValid() ):
            return False
            
        editor = QComboBox(parent)
        editor.addItems( settings.enabledDepartments() )
        editor.setCurrentIndex( editor.findText( index.data( Qt.DisplayRole ).toString() ) )
        return editor
    
    def setModelData( self, editor, model, index ):
        """
                Switches the data for the inputed index to the given
                department
                
                :param      editor:
                :type       <QWidget>
                
                :param      model:
                :type       <QAbstractItemModel>:
                
                :param      index:
                :type       <QModelIndex>:
        """
        model.setData( index, QVariant(editor.currentText()), Qt.DisplayRole )
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

