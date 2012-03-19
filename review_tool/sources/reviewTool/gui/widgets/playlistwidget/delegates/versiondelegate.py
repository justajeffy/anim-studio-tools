##
#   \namespace  reviewTool.gui.widgets.playlistwidget.delegates.versiondelegate
#
#   \remarks    Defines the VersionDelegate class used when rendering version information
#               in the playlistwidget
#   
#   \author     Dr. D Studios
#   \date       08/09/11
#

from PyQt4.QtCore               import  QVariant,\
                                        Qt

from PyQt4.QtGui                import  QApplication,\
                                        QColor,\
                                        QComboBox

from ....delegates.griddelegate import  GridDelegate
from .....kernel                import  core
from .....                      import  resources

class VersionDelegate(GridDelegate):
    def __init__( self, parent ):
        super(VersionDelegate,self).__init__( parent )
        
        self._latestColor   = QColor('green')
        self._outdatedColor = QColor('red')
    
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
        
        editor  = QComboBox(parent)
        editor.addItems( self.options(index) )
        editor.setCurrentIndex( editor.findText( index.data( Qt.DisplayRole ).toString() ) )
        return editor
    
    def drawOverlay( self, painter, option, index ):
        """
                Paints the overlay color for this item if the current version
                based on whether or not it is the latest version available
                
                :param      painter:
                :type       <QPainter>:
                
                :param      option:
                :param      <QStyleOptionItem>:
                
                :param      index:
                :type       <QModelIndex>:
        """
        # extract the option's frame range information
        version = index.data( Qt.DisplayRole ).toString()
        latest  = self.isLatest(index)
        
        # determine the look for various options based on the latest result check
        if ( latest == -1 ):
            toolTip = 'No data cached to compare for latest.'
            color   = None
        
        # show the outdated version information
        elif ( latest == 0 ):
            toolTip = '%s is not the latest Version (based on sort order)' % version
            color   = self.outdatedColor()
        
        # show the latest version information
        else:
            toolTip = '%s is the latest Version (based on sort order)' % version
            color   = self.latestColor()
            
        
        # update the tool tip for the index
        model   = index.model()
        blocked = model.signalsBlocked()
        model.blockSignals(True)
        model.setData( index, QVariant(toolTip), Qt.ToolTipRole )
        model.blockSignals(blocked)
        
        if ( not color ):
            return
        
        # create the out of range color
        color.setAlpha(100)
        
        # draw the overlay
        painter.setPen( Qt.NoPen )
        painter.setBrush(color)
        painter.drawRect( option.rect )
    
    def isLatest( self, index ):
        """
                Checks to see if the inputed index's data
                is currently set to the latest possible version
                based on the cached data
                
                :param      index:
                :type       <QModelIndex:
                
                :return     <int>: 1 if True, 0 if False, -1 if not enough data
        """
        clip = index.data( Qt.UserRole ).toPyObject()
        
        # make sure we have a valid list of options
        if ( not clip ):
            return -1
        
        options = clip.siblingNames()
        if ( not options ):
            return -1
        
        # make sure the current version is in the list
        version = str(index.data( Qt.DisplayRole ).toString())
        
        # make sure the version is the first item in the list
        return int(options[0] == version)
    
    def latestColor( self ):
        """
                Returns the color to be used when
                drawing a version is the latest option
                
                :return     <QColor>:
        """
        return self._latestColor
    
    def options( self, index ):
        """
                Collects a list of options for the versions for the inputed index
                
                :param      index:
                :type       <QModelIndex:
                
                :return     <list> [ <str>, .. ]:
        """
        clip = index.data( Qt.UserRole ).toPyObject()
        if ( not clip ):
            return []
        
        # load cached data
        names = clip.siblingNames()
        if ( names ):
            return names
        
        # create cached data
        QApplication.setOverrideCursor( Qt.WaitCursor )
        core.info( 'Loading versions for %s...' % index.data( Qt.DisplayRole ).toString() )
        clip.entity().collectVersions()
        names = clip.siblingNames()
        QApplication.restoreOverrideCursor()
            
        return names
    
    def outdatedColor( self ):
        """
                Returns the color to be used when
                drawing a version is the oudated option
                
                :return     <QColor>:
        """
        return self._outdatedColor
    
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
        super(VersionDelegate,self).paint( painter, option, index )
        
        # force the background to be drawn
        self.drawOverlay( painter, option, index )
    
    def setLatestColor( self, color ):
        """
                Sets the latest to be used when drawing a
                version that is the first selected option
                
                :return     <QColor>:
        """
        self._latestColor = QColor(color)
    
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
        
    def setOutdatedColor( self, color ):
        """
                Sets the color to be used when drawing
                a version that is not the first selected
                option
                
                :return     <QColor>:
        """
        self._outdatedColor = QColor(color)
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

