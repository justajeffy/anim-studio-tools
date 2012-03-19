##
#   \namespace  reviewTool.gui.contextwidget.versionitemwidget
#
#   \remarks    Creates a widget to be used in the VersionSelectDialog when a user picks versions from the UI
#   
#   \author     Dr. D Studios
#   \date       07/28/11
#

import os.path
import subprocess

import PyQt4.uic

from PyQt4.QtCore   import  Qt
from PyQt4.QtGui    import  QColor,\
                            QFontMetrics,\
                            QTextDocument,\
                            QWidget

from ....           import resources

class VersionItemWidget( QWidget ):
    def __init__( self, parent, item, version ):
        # initialize the super class
        QWidget.__init__( self, parent )
        
        # load the ui
        uifile = os.path.join(os.path.dirname(__file__),'ui/%s.ui' % os.path.basename(__file__).split('.')[0])
        PyQt4.uic.loadUi( uifile, self )
        
        # set the version
        self.uiActiveCHK.setChecked(    version.isActive() )
        self.uiNameLBL.setText(         version.name() )
        self.uiDateLBL.setText(         version.createdAt().strftime( '%d/%m/%y' ) )
        self.uiCommentsLBL.setToolTip(  version.comments() )
        
        self.uiStatusLBL.setPixmap(     resources.pixmap( 'img/render_status/%s.png' % version.status() ) )
        self.uiInfoBTN.setIcon(         resources.icon( 'img/context/info.png' ) )
        
        self.uiInfoBTN.clicked.connect( self.showInfo )
        
        # create custom properties
        self._versionItem       = item
        self._version           = version
        self._highlightBrush    = self.palette().color( self.palette().Highlight )
        self._baseBrush         = None
        
    def enterEvent( self, event ):
        super(VersionItemWidget,self).enterEvent(event)
        
        # udpate the palette and fill
        if ( self._baseBrush == None ):
            self._baseBrush = self._versionItem.background(0)
            
        self._versionItem.setBackground( 0, self._highlightBrush )
    
    def isActive( self ):
        return self.uiActiveCHK.isChecked()
    
    def leaveEvent( self, event ):
        super(VersionItemWidget,self).leaveEvent(event)
        self._versionItem.setBackground( 0, self._baseBrush )
    
    def resizeEvent( self, event ):
        super(VersionItemWidget,self).resizeEvent(event)
        
        # make sure we have finished initializing before we resize
        if ( not '_version' in self.__dict__ ):
            return
            
        metrics = QFontMetrics(self.font())
        version = self._version
        
        # update the labels with eliding based on new spacing
        self.uiUserLBL.setText(         metrics.elidedText( version.username(),                     Qt.ElideRight, self.uiUserLBL.width() ) )
        
        # convert out any html data
        comments = str(version.comments())
        doc = QTextDocument()
        doc.setHtml(comments)
        comments = str(doc.toPlainText())
        
        self.uiCommentsLBL.setText(     metrics.elidedText( comments.replace('\n',' '),   Qt.ElideRight, self.uiCommentsLBL.width() ) )
    
    def showInfo( self ):
        self._version.explore()
    
    def version( self ):
        return self._version
        
    def toggleActive( self ):
        self.uiActiveCHK.setChecked(not self.uiActiveCHK.isChecked())
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

