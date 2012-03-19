##
#   \namespace  reviewTool.gui.widgets.playlistwidget.delegates.framedelegate
#
#   \remarks    Defines the FrameDelegate class used when rendering frame information
#               in the playlistwidget
#   
#   \author     Dr. D Studios
#   \date       08/09/11
#

from PyQt4.QtCore               import  QVariant,\
                                        Qt

from PyQt4.QtGui                import  QColor

from ....delegates.griddelegate import GridDelegate

class FrameDelegate(GridDelegate):
    def __init__( self, parent ):
        super(FrameDelegate,self).__init__( parent )
        
        self._outOfRangeColor = QColor('red')
    
    def drawDisplay( self, painter, option, rect, text ):
        """
                Strips out any commas from the text when drawing the display
                
                :param      painter:
                :type       <QPainter>:
                
                :param      option:
                :type       <QStyleOptionViewItem>:
                
                :param      rect:
                :type       <QRect>:
                
                :param      text:
                :type       <QString>:
        """
        text = str(text).replace(',','')
        super(FrameDelegate,self).drawDisplay( painter, option, rect, text )
    
    def drawOverlay( self, painter, option, index ):
        """
                Paints the overlay color for this item if the current frame
                is out of range
                
                :param      painter:
                :type       <QPainter>:
                
                :param      option:
                :param      <QStyleOptionItem>:
                
                :param      index:
                :type       <QModelIndex>:
        """
        # extract the option's frame range information
        currFrame   = index.data( Qt.EditRole ).toInt()[0]
        frameRange  = index.data( Qt.UserRole ).toPyObject()
        
        # determine if the current frame is within the valid range
        if ( frameRange and currFrame < frameRange[0] ):
            toolTip = '%s is below the minimum frame (%s)' % (currFrame,frameRange[0])
        elif ( frameRange and frameRange[1] < currFrame ):
            toolTip = '%s is above the maximum frame (%s)' % (currFrame,frameRange[1])
        else:
            toolTip = ''
        
        # update the tool tip for the index
        model   = index.model()
        blocked = model.signalsBlocked()
        model.blockSignals(True)
        model.setData( index, QVariant(toolTip), Qt.ToolTipRole )
        model.blockSignals(blocked)
        
        if ( not toolTip ):
            return
        
        # create the out of range color
        color = QColor(self.outOfRangeColor())
        color.setAlpha(100)
        
        # draw the overlay
        painter.setPen( Qt.NoPen )
        painter.setBrush(color)
        painter.drawRect( option.rect )
    
    def outOfRangeColor( self ):
        """
                Returns the out-of-range color to be used when
                drawing a frame that is out of its index's range
                
                :return     <QColor>:
        """
        return self._outOfRangeColor
    
    def paint( self, painter, option, index ):
        """
                Overloads the base paint method for a QItemDelegate
                to force the overlay color options to be drawn.
                
                :param      painter:
                :type       <QPainter>:
                
                :param      option:
                :type       <QStyleOptionViewItem>:
                
                :param      index:
                :type       <QModelIndex>:
        """
        super(FrameDelegate,self).paint( painter, option, index )
        
        # force the background to be drawn
        self.drawOverlay( painter, option, index )
    
    def setOutOfRangeColor( self, color ):
        """
                Sets the out-of-range color to be used when drawing a
                frame that is out of it's index's range
                
                :return     <QColor>:
        """
        self._outOfRangeColor = QColor(color)
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

