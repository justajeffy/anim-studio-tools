##
#   \namespace  reviewTool.gui.contextwidget.contextversionswidget
#
#   \remarks    Creates a widget to be used in each of the entity departments
#   
#   \author     Dr. D Studios
#   \date       07/27/11
#

import os

import PyQt4.uic

from PyQt4.QtCore   import  Qt
from PyQt4.QtGui    import  QFontMetrics,\
                            QWidget

from .... import resources

class ContextVersionsWidget( QWidget ):
    def __init__( self, contextWidget, entity, versions ):
        # initialize the super class
        super(ContextVersionsWidget,self).__init__( contextWidget )
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'ui/contextversionswidget.ui' )
        PyQt4.uic.loadUi( uifile, self )
        
        # set the icons
        self.uiMoreBTN.setIcon( resources.icon('img/context/arrow_down.png') )
        
        # setup custom properties
        self._contextWidget     = contextWidget
        self._entity            = entity
        self._versions          = versions
        
        # create connections
        self.uiActiveCHK.clicked.connect(   self.setActive )
        self.uiMoreBTN.clicked.connect(     self.pickVersions )
        
        # refresh the interface
        self.refresh()
    
    def activeVersions( self ):
        return self._activeVersions
    
    def contextWidget( self ):
        return self._contextWidget
    
    def currentVersions( self ):
        if ( self.isActive() ):
            return self.activeVersions()
        else:
            return [self._versions[0]]
    
    def emitActiveStateChanged( self ):
        self.contextWidget().emitActiveStateChanged(self.versions())
    
    def isActive( self ):
        return self.uiActiveCHK.isChecked()
    
    def pickVersions( self ):
        # check to see if anything changes
        active_cache = self._activeVersions
        
        # prompt the user to edit the version selection
        self.contextWidget().pickVersions( self._versions )
        
        # check the new active versions
        new_active = [ version for version in self._versions if version.isActive() ]
        
        # check the new active cache state
        if ( set(active_cache) != set(new_active) ):
            self.refresh()
            self.emitActiveStateChanged()
    
    def refresh( self ):
        # determine the current active versions
        versions                = [ version for version in self._versions if version.isActive() ]
        
        # make sure we have a fresh list that isn't affected from below
        self._activeVersions    = list(versions)
        
        # if there are no active versions, reflect that
        # in the ui, and show data for the latest version
        if ( not versions ):
            versions = [self._versions[0]]
            self.uiActiveCHK.setChecked(False)
        else:
            self.uiActiveCHK.setChecked(True)
        
        curr_status     = 'none'
        version_names   = []
        
        # show the different
        for version in versions:
            version_names.append(version.name())
            
            if ( curr_status == 'none' ):
                curr_status = version.status()
            elif ( curr_status != version.status() ):
                curr_status = 'multi'
        
        # update the information
        self.uiStatusLBL.setPixmap( resources.pixmap('img/render_status/%s.png' % curr_status ) )
        self.uiVersionsLBL.setText( ','.join(version_names) )
    
    def setActive( self, state = True ):
        if ( not state ):
            # if the current state is not active
            for version in self._activeVersions:
                version.setActive(False)
        else:
            self._versions[0].setActive(True)
        
        self.refresh()
        self.emitActiveStateChanged()
    
    def setActiveVersions( self, versions ):
        for version in self.versions():
            version.setActive( version in versions )
        
        self.refresh()
    
    def setVersions( self, versions ):
        self._versions = versions
        self.refresh()
    
    def versions( self ):
        return self._versions
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

