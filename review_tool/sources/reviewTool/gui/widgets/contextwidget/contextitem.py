##
#   \namespace  reviewTool.gui.widgets.contextwidget.contextitem
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

from PyQt4.QtCore       import  QSize
from PyQt4.QtGui        import  QApplication,\
                                QTreeWidgetItem

from .contextentityitem import ContextEntityItem

#--------------------------------------------------------------------------------

class ContextItem( QTreeWidgetItem ):
    def __init__( self, context ):
        super(ContextItem,self).__init__( [context.name()] )
        self.setSizeHint( 0, QSize( 0, 18 ) )
        
        self._context = context
        self._loaded = False
    
    def context( self ):
        """
        Returns the context instance that the item represents
        
        :return     <Context>:
        """
        return self._context
    
    def isLoading( self ):
        return False
    
    def loadEntities( self, focusEntity = None ):
        """
                Loads the various context entities that can be seen for
                this context instance
                
                :param  focusEntity:        entity to focus on
                :type   <str> || None:
                
        """
        tree = self.treeWidget()
        if ( not tree ):
            return
        
        if ( self._loaded ):
            return
        
        self._loaded    = True
        first           = True
        
        geom            = tree.geometry()
        
        # load the different contexts
        for context in self.context().entities():
            if ( first ):
                self.setExpanded(True)
                first = False
                
            item = ContextEntityItem(context)
            self.addChild(item)
            
            if ( item.text(0) == focusEntity ):
                tree.setCurrentItem(item)
                
            item.initData()

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

