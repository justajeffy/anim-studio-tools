#         Dr. D Studios - Software Disclaimer
#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

__authors__ = list(set(["eric.hulser","Eric Hulser"]))
__version__ = '18216'
__revision__ = __version__ # For pylint
__date__ = 'May 13 2011 3:08:59 PM'

__copyright__ = '2011'
__license__ = "Copyright 2011 Dr D Studios Pty Limited"
__contact__ = "eric.hulser@drdstudios.com"
__status__ = "Development"
__doc__ = """   """

from PyQt4.QtGui import QDialog

class StatusPrefsDialog( QDialog ):
    _instance = None
    
    def __init__( self, parent ):
        # initialize the super class
        QDialog.__init__( self, parent )
        
        # load the ui
        import os.path
        import PyQt4.uic
        PyQt4.uic.loadUi( os.path.join(os.path.dirname(__file__),'resource/statusprefsdialog.ui'), self )
        
        # setup the proper stretching we want
        header = self.uiStatusTREE.header()
        header.setResizeMode( 0, header.Stretch )
        header.setResizeMode( 1, header.ResizeToContents )
        
        # create custom properties
        self._statusHash    = {}
        self._statusPrefs   = None
        self._loaded        = False
        
        # safe-bind the dropEvent method for the tree widget (not handled by eventFilter in Qt)
        import types
        bound_method = types.MethodType( self.handleDropEvent.im_func, self.uiStatusTREE, self.uiStatusTREE.__class__ )
        self.uiStatusTREE.__dict__[ 'dropEvent' ] = bound_method
        
        # create connections
        self.uiDialogBTNS.accepted.connect( self.accept )
        self.uiDialogBTNS.rejected.connect( self.reject )
    
    def accept( self ):
        QDialog.accept( self )
        
        # record the preferences from the ui
        from PyQt4.QtCore import Qt
        data = {}
        for i in range( self.uiStatusTREE.topLevelItemCount() ):
            item = self.uiStatusTREE.topLevelItem(i)
            status = str(item.text(1))
            status_data = {}
            status_data['enabled']  = True if item.checkState(0) == Qt.Checked else False
            status_data['order']    = i
            data[status] = status_data
        
        # save the data to the prefs
        self._statusPrefs.set_data( data )
    
    def handleDropEvent( tree, event ):
        source = tree.currentItem()
        target = tree.itemAt(event.pos())
        
        if ( source and target and source != target ):
            # pull the source out of the tree
            tree.takeTopLevelItem(tree.indexOfTopLevelItem(source))
            
            # insert the item back into the tree
            tree.insertTopLevelItem(tree.indexOfTopLevelItem(target),source)
            
            # reset the current item as the dropped item
            tree.setCurrentItem(source)
    
    def refresh( self ):
        self.uiStatusTREE.blockSignals(True)
        self.uiStatusTREE.setUpdatesEnabled(False)
        
        self.uiStatusTREE.clear()
        
        # collect the data from the prefs
        ordered = self._statusPrefs.values('order').items()
        
        # sort the items by their order
        ordered.sort( lambda x,y: cmp( x[1], y[1] ) )
        
        # add the statuses to the tree
        from PyQt4.QtCore   import Qt
        from PyQt4.QtGui    import QTreeWidgetItem, QPalette, QIcon
        
        for status, order in ordered:
            # grab info from the hash
            icon, display_name = self._statusHash.get(status,('','Missing'))
            
            item = QTreeWidgetItem([display_name,status])
            item.setIcon( 0, QIcon(icon) )
            
            # check the item if it is enabled
            enabled = self._statusPrefs.value(status,'enabled')
            item.setCheckState( 0, Qt.Checked if enabled else Qt.Unchecked )
            
            # set some ui options for the status name to hide it a bit
            item.setTextAlignment( 1, Qt.AlignRight | Qt.AlignVCenter )
            item.setForeground( 1, self.uiStatusTREE.palette().color(QPalette.Base).darker(140) )
            
            # add to the tree
            self.uiStatusTREE.addTopLevelItem(item)
            
        self.uiStatusTREE.blockSignals(False)
        self.uiStatusTREE.setUpdatesEnabled(True)
    
    def setStatusHash( self, statusHash ):
        self._statusHash = statusHash
    
    def setStatusPrefs( self, statusPrefs ):
        self._statusPrefs = statusPrefs
    
    @staticmethod
    def edit( parent, status_hash, status_prefs, draggable = True, tip = '' ):
        inst = StatusPrefsDialog._instance
        
        # cache the status prefs dialog
        if ( not inst ):
            inst = StatusPrefsDialog(parent)
            inst.setPalette(parent.tree.palette())
        
        if ( not draggable ):
            inst.uiStatusTREE.setDragDropMode( inst.uiStatusTREE.NoDragDrop )
        if ( tip ):
            inst.label.setText( tip )
            
        inst.setStatusHash( status_hash )
        inst.setStatusPrefs( status_prefs )
        inst.refresh()
        
        if ( inst.exec_() ):
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

