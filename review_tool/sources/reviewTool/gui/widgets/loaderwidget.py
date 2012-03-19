##
#   \namespace  reviewTool.gui.loaderwidget
#
#   \remarks    Creates an AJAX loading widget
#   
#   \author     Dr. D Studios
#   \date       07/28/11
#

import os.path
import PyQt4.uic

from PyQt4.QtCore   import  QByteArray,\
                            Qt,\
                            QEvent
                            
from PyQt4.QtGui    import  QColor,\
                            QLabel,\
                            QMovie,\
                            QPalette,\
                            QVBoxLayout,\
                            QSizePolicy,\
                            QWidget

from ...            import resources

class LoaderWidget( QWidget ):
    MOVIE = None
    
    def __init__( self, parent ):
        # initialize the super class
        super(LoaderWidget,self).__init__( parent )
        
        # create the movie
        if ( LoaderWidget.MOVIE == None ):
            LoaderWidget.MOVIE = QMovie(resources.find('img/main/ajax-loader.gif'))
            LoaderWidget.MOVIE.start()
        
        # create the movie label
        self._movieLabel    = QLabel(self)
        self._movieLabel.setMovie(LoaderWidget.MOVIE)
        self._movieLabel.setAlignment(Qt.AlignCenter)
        
        self._messageLabel  = QLabel(self)
        self._messageLabel.setAlignment(Qt.AlignCenter)
        
        palette = self._messageLabel.palette()
        palette.setColor( palette.WindowText, QColor('gray') )
        self._messageLabel.setPalette(palette)
            
        # create the interface
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addStretch()
        layout.addWidget( self._movieLabel )
        layout.addWidget( self._messageLabel )
        layout.addStretch()
        
        self.setLayout(layout)
        self.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Preferred )
        self.resize(0,0)
        self.setMessage('')
        
        # set the default properties
        self.setAutoFillBackground(True)
        self.setBackgroundRole( QPalette.Window )
        
        # create custom properties
        clr = QColor('black')
        clr.setAlpha(150)
        self._backgroundColor = clr
    
    def backgroundColor( self ):
        return self._backgroundColor
    
    def eventFilter( self, object, event ):
        if ( object == self.parent() and event.type() == QEvent.Resize ):
            self.resize(object.size())
        return False
    
    def message( self ):
        return self._messageLabel.text()
    
    def paintEvent( self, event ):
        # make sure we have the proper palette, as if this widget is
        # added to a hierarchy, it may overwrite whats there
        palette = self.palette()
        palette.setColor( QPalette.Window, self._backgroundColor )
        self.setPalette(palette)
        
        super(LoaderWidget,self).paintEvent(event)
    
    def setBackgroundColor( self, clr ):
        self._backgroundColor = QColor(clr)
    
    def setMessage( self, message ):
        self._messageLabel.setText(message)
        self._messageLabel.setVisible(message != '')
    
    @staticmethod
    def start( widget, message = '' ):
        loaders = widget.findChildren(LoaderWidget)
        if ( loaders ):
            loader = loaders[0]
        else:
            loader = LoaderWidget(widget)
            widget.installEventFilter(loader)
            loader.resize(widget.size())
            loader.show()
        
        loader.setMessage(message)
        
        return loader
    
    @staticmethod
    def stop(widget):
        for loader in widget.findChildren(LoaderWidget):
            loader.close()
            loader.setParent(None)
            loader.deleteLater()
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

