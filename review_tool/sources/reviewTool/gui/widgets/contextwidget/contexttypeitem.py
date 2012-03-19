##
#   \namespace  reviewTool.gui.widgets.contextwidget.contexttypeitem
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

from PyQt4.QtCore   import  QSize,\
                            Qt

from PyQt4.QtGui    import  QColor,\
                            QTreeWidgetItem

from .contextitem   import ContextItem

class ContextTypeItem( QTreeWidgetItem ):
    def __init__( self, name, contextType ):
        super(ContextTypeItem,self).__init__( [name] )
        
        # define custom properties
        self.setFlags( Qt.ItemIsEnabled )
        clr = QColor('white')
        clr.setAlpha(150)
        self.setBackground( 0, clr )
        self.setSizeHint( 0, QSize( 0, 18 ) )
        
        self._contextType = contextType
    
    def contextType( self ):
        """
        Returns the context type class that this item represents
        
        :return     <subclass of Context>
        """
        return self._contextType
    
    def loadContexts( self ):
        """
        Reloads the items for this context item
        """
        # clear the current children
        while ( self.childCount() ):
            self.takeChild(0)
        
        # load the different contexts
        for context in self.contextType().contexts():
            self.addChild(ContextItem(context))
        
        self.setExpanded(True)
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

