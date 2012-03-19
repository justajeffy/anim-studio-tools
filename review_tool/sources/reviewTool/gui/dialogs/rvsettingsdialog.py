##
#   \namespace  reviewTool.gui.dialogs.rvsettingsdialog
#
#   \remarks    Defines a dialog for manipulating common rv_tools settings from Review Tool
#   
#   \author     Dr. D Studios
#   \date       08/03/11
#

import os.path

import PyQt4.uic
from PyQt4.QtGui import QDialog, QFileDialog

import rv_tools.settings

class RVSettingsDialog( QDialog ):
    def __init__( self, parent = None ):
        super(RVSettingsDialog,self).__init__(parent)
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'ui/rvsettingsdialog.ui' )
        PyQt4.uic.loadUi( uifile, self )
        
        # initialize ui
        self.uiProfileDDL.addItems(rv_tools.settings.profileNames())
        
        # create connections
        self.uiLocalPrefixBTN.clicked.connect( self.pickButton )
        
        # resore settings
        self.restoreSettings()
    
    def accept( self ):
        # save settings
        self.recordSettings()
        
        super(RVSettingsDialog,self).accept()
    
    def pickButton( self ):
        filepath = QFileDialog.getExistingDirectory( self, 'Select Root Path', self.uiLocalPrefixTXT.text() )
        if ( filepath ):
            self.uiLocalPrefixTXT.setText(filepath)
    
    def recordSettings( self ):
        rv_tools.settings.setValue('localize',          self.uiLocalPathsCHK.isChecked())
        rv_tools.settings.setValue('local_prefix',      str(self.uiLocalPrefixTXT.text()))
        rv_tools.settings.setValue('separate_process',  self.uiSubprocessCHK.isChecked())
        rv_tools.settings.setValue('executable',        str(self.uiExecDDL.currentText()))
        rv_tools.settings.setValue('current_profile',   str(self.uiProfileDDL.currentText()))
        rv_tools.settings.setValue('use_dpx_format',    self.uiConvertToDpxCHK.isChecked())
    
    def restoreSettings( self ):
        # restore the settings from the module
        self.uiLocalPathsCHK.setChecked(    rv_tools.settings.value('localize'))
        self.uiLocalPrefixTXT.setText(      rv_tools.settings.value('local_prefix'))
        self.uiSubprocessCHK.setChecked(    rv_tools.settings.value('separate_process'))
        self.uiConvertToDpxCHK.setChecked(  rv_tools.settings.value('use_dpx_format'))
        
        self.uiExecDDL.setCurrentIndex(     self.uiExecDDL.findText(    rv_tools.settings.value('executable')))
        self.uiProfileDDL.setCurrentIndex(  self.uiProfileDDL.findText( rv_tools.settings.value('current_profile')))
        
    # define static methods
    @staticmethod
    def edit( parent ):
        dlg = RVSettingsDialog( parent )
        if ( dlg.exec_() ):
            return True
        return False
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

