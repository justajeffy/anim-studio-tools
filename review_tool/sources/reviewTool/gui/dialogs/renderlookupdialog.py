##
#   \namespace  reviewTool.gui.dialogs.renderlookupdialog
#
#   \remarks    Creates a render lookup system for finding versions based on the render type and shared source
#   
#   \author     eric.hulser@drdstudios.com
#   \author     Dr. D Studios
#   \date       08/12/11
#

import os.path

import PyQt4.uic

from PyQt4.QtCore   import  Qt
from PyQt4.QtGui    import  QApplication,\
                            QDialog,\
                            QTableWidgetItem

from ...            import resources

#--------------------------------------------------------------------------------

class VersionItem( QTableWidgetItem ):
    def __init__( self, version ):
        super(VersionItem,self).__init__( version.displayName() )
        
        self.setCheckState( Qt.Unchecked )
        
        self._version = version
    
    def version( self ):
        return self._version
        
#--------------------------------------------------------------------------------

class RenderLookupDialog( QDialog ):
    def __init__( self, parent = None ):
        super(RenderLookupDialog,self).__init__(parent)
        
        self.setPalette(parent.palette())
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'ui/renderlookupdialog.ui' )
        PyQt4.uic.loadUi( uifile, self )
        
        # define custom properties
        self._baseVersions = []

        # create connections
        self.uiDialogBTNS.accepted.connect( self.accept )      # assumes there is a uiDialogBTNS in the ui file
        self.uiDialogBTNS.rejected.connect( self.reject )      # assumes there is a uiDialogBTNS in the ui file
        
    # define instance methods
    def currentVersions( self ):
        """
                Returns all the currently checked items from the ui
                
                :return    <list> [ <Version>, ]:
        """
        output = []
        
        for r in range(self.uiVersionTABLE.rowCount()):
            for c in range(self.uiVersionTABLE.columnCount()):
                item = self.uiVersionTABLE.item(r,c)
                
                if ( not item ):
                    continue
                
                if ( item.checkState() == Qt.Checked ):
                    output.append(item.version())
        
        return output

    def setBaseVersions( self, baseVersions ):
        """
                Loads the table with the base versions, loading related versions from
                the api
                
                :param     baseVersions:
                :type       <list> [ <Version>, .. ]:
        """
        self._baseVersions = baseVersions
        
        QApplication.setOverrideCursor( Qt.WaitCursor )
        
        self.uiVersionTABLE.blockSignals(True)
        self.uiVersionTABLE.setUpdatesEnabled(False)
        
        # clear the table
        self.uiVersionTABLE.clear()
        self.uiVersionTABLE.setColumnCount( len(self._baseVersions) )
        self.uiVersionTABLE.setHorizontalHeaderLabels( [ baseVersion.displayName() for baseVersion in self._baseVersions ] )
        self.uiVersionTABLE.verticalHeader().hide()
        
        # load the header columns
        header  = self.uiVersionTABLE.horizontalHeader()
        header.setMinimumSectionSize( 200 )
        for c in range(self.uiVersionTABLE.columnCount()):
            header.setResizeMode( c, header.Stretch )
        
        # load the items
        column = 0
        for version in baseVersions: 
            row         = 0
            reviewType  = version.reviewType()
            
            for sibling in version.siblingsOfCommonDescent():
                # make sure the sibling is of a different type
                if ( sibling.reviewType() == reviewType ):
                    continue
                
                # insert a new row for the sibling
                if ( row == self.uiVersionTABLE.rowCount() ):
                    self.uiVersionTABLE.insertRow(row)
                
                item = VersionItem(sibling)
                item.setCheckState( Qt.Checked if not row else Qt.Unchecked )
                self.uiVersionTABLE.setItem( row, column, item )
                
                row += 1
            column += 1
        
        self.uiVersionTABLE.blockSignals(False)
        self.uiVersionTABLE.setUpdatesEnabled(True)
        
        QApplication.restoreOverrideCursor()

    # define static methods
    @staticmethod
    def collectVersions( versions, parent = None ):
        dlg = RenderLookupDialog( parent )
        dlg.setBaseVersions( versions )
        if ( dlg.exec_() ):
            return dlg.currentVersions()
        return []
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

