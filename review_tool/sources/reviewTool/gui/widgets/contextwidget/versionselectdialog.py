##
#   \namespace  reviewTool.gui.contextwidget.versionsselectdialog
#
#   \remarks    Creates a popup for managing version selections
#   
#   \author     Dr. D Studios
#   \date       07/28/11
#

from PyQt4.QtCore       import Qt, QSize

from PyQt4.QtGui        import  QDialog,\
                                QTreeWidget,\
                                QTreeWidgetItem,\
                                QVBoxLayout

from .versionitemwidget import VersionItemWidget
from ....               import resources

class VersionItem( QTreeWidgetItem ):
    def __init__( self, tree, version ):
        super(VersionItem,self).__init__()
        
        self._versionWidget = VersionItemWidget(tree,self,version)
    
    def versionWidget( self ):
        return self._versionWidget
        
class VersionSelectDialog( QDialog ):
    def __init__( self, parent ):
        # initialize the super class
        super( VersionSelectDialog, self ).__init__( parent )
        
        # create the tree
        self.uiVersionsTREE = QTreeWidget(self)
        self.uiVersionsTREE.setAlternatingRowColors(True)
        self.uiVersionsTREE.setRootIsDecorated(False)
        self.uiVersionsTREE.setSelectionMode( self.uiVersionsTREE.NoSelection )
        
        header = self.uiVersionsTREE.header()
        header.setVisible(False)
        
        # create the layout
        layout = QVBoxLayout()
        layout.addWidget(self.uiVersionsTREE)
        layout.setContentsMargins(0,0,0,0)
        
        # inherit the highlight palette
        palette = self.palette()
        palette.setColor(palette.Highlight,parent.palette().color(palette.Highlight))
        self.setPalette(palette)
        
        # set dialog information
        self.setLayout(layout)
        self.setWindowFlags( Qt.Popup )
        self.resize(500,250)
        
        # create connections
        self.uiVersionsTREE.itemClicked.connect( self.acceptItem )
    
    def closeEvent( self, event ):
        # update all the items for this
        for i in range( self.uiVersionsTREE.topLevelItemCount() ):
            item    = self.uiVersionsTREE.topLevelItem(i)
            widget  = item.versionWidget()
            version = widget.version()
            
            # match the active state
            version.setActive(widget.isActive())
        
        super(VersionSelectDialog,self).closeEvent(event)
    
    def acceptItem( self, item ):
        # handle version change information
        widget = item.versionWidget()
        widget.toggleActive()
        
        # accept the dialog
        self.close()
    
    def popup( self, versions ):
        self.uiVersionsTREE.setUpdatesEnabled(False)
        self.uiVersionsTREE.blockSignals(True)
        
        self.uiVersionsTREE.clear()
        for version in versions:
            item = VersionItem(self.uiVersionsTREE, version)
            self.uiVersionsTREE.addTopLevelItem( item )
            self.uiVersionsTREE.setItemWidget( item, 0, item.versionWidget() )
        
        # reset the scrolling
        self.uiVersionsTREE.verticalScrollBar().setValue(0)
        
        self.uiVersionsTREE.setUpdatesEnabled(True)
        self.uiVersionsTREE.blockSignals(False)
        
        return self.exec_()
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

