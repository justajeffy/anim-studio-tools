##
#   \namespace  reviewTool.gui.delegates.griddelegate
#
#   \remarks    [desc::commented]
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

from PyQt4.QtCore   import QLine
from PyQt4.QtGui    import QItemDelegate

#--------------------------------------------------------------------------------

class GridDelegate( QItemDelegate ):
    def __init__( self, parent ):
        super(GridDelegate,self).__init__(parent)
        
        self._borderColor   = parent.palette().color(parent.palette().Mid)
        self._showColumns   = True
        self._showRows      = True
    
    def borderColor( self ):
        return self._borderColor
    
    def drawGrid( self, painter, option, index ):
        
        # grab the rect information
        left    = option.rect.left()
        right   = option.rect.right()
        top     = option.rect.top()
        bottom  = option.rect.bottom()
        
        # draw to the root for the first item
        if ( not index.column() ):
            left = 0
        
        lines = []
        if ( self.showColumns() ):
            # only need to draw the right most line, since items will build together
            lines.append( QLine(right,top,right,bottom) )
        if ( self.showRows() ):
            # only need to draw the bottom most line, since items will draw together
            lines.append( QLine(left,bottom,right,bottom) )
        
        painter.setPen(self.borderColor())
        painter.drawLines(lines)
    
    def paint( self, painter, option, index ):
        super(GridDelegate,self).paint(painter,option,index)
        
        # draw the grid
        self.drawGrid(painter,option,index)
    
    def setBorderColor( self, clr ):
        self._borderColor = QColor(clr)
    
    def setShowColumns( self, state = True ):
        self._showColumns = state
    
    def setShowRows( self, state = True ):
        self._showRows = state
    
    def showColumns( self ):
        return self._showColumns
    
    def showRows( self ):
        return self._showRows

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

