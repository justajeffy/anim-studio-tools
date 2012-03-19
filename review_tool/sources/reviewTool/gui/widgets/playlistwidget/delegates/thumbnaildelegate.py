##
#   \namespace  reviewTool.gui.widgets.playlistwidget.delegates.thumbnaildelegate
#
#   \remarks    Defines the ThumbnailDelegate class used when rendering thumbnail information
#               in the playlistwidget
#   
#   \author     Dr. D Studios
#   \date       08/09/11
#

from PyQt4.QtCore               import  Qt

from PyQt4.QtGui                import  QPixmap,\
                                        QPixmapCache

from ....delegates.griddelegate import  GridDelegate
from .....                      import  settings
from .....                      import  resources

# up the cache limit since this is a graphics heavy app
QPixmapCache.setCacheLimit( 10240 * 48 ) # in kilobytes

class ThumbnailDelegate(GridDelegate):
    def drawOverlay( self, painter, option, index ):
        """
                Paints the overlay color for this item if the current thumbnail
                based on whether or not it is the latest thumbnail available
                
                :param      painter:
                :type       <QPainter>:
                
                :param      option:
                :param      <QStyleOptionItem>:
                
                :param      index:
                :type       <QModelIndex>:
        """
        # extract the option's frame range information
        key     = index.data( Qt.UserRole ).toString()
        if ( not key ):
            return
        
        pixmap  = QPixmap()
        
        # restore the pixmap from cache
        if ( not QPixmapCache.find(key,pixmap) ):
            pixmap.load(settings.iconCachePath( '%s.jpg' % key ))
            QPixmapCache.insert(key,pixmap)
            
        # draw the overlay
        painter.drawPixmap( option.rect, pixmap )
        
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
        self.drawBackground( painter, option, index )
        
        # extract the option's frame range information
        key     = index.data( Qt.UserRole ).toString()
        if ( not key ):
            return
        
        pixmap  = QPixmap()
        
        # restore the pixmap from cache
        if ( not QPixmapCache.find(key,pixmap) ):
            pixmap.load(settings.iconCachePath( '%s.jpg' % key ))
            QPixmapCache.insert(key,pixmap)
            
        # draw the overlay
        painter.drawPixmap( option.rect, pixmap )
    
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

