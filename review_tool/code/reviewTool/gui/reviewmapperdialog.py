#         Dr. D Studios - Software Disclaimer
#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

__authors__ = list(set(["eric.hulser","Eric Hulser"]))
__version__ = '18216'
__revision__ = __version__ # For pylint
__date__ = 'Jun 8 2011 11:05:51 AM'

__copyright__ = '2011'
__license__ = "Copyright 2011 Dr D Studios Pty Limited"
__contact__ = "eric.hulser@drdstudios.com"
__status__ = "Development"
__doc__ = """   """

import os.path

import PyQt4.uic

from PyQt4.QtCore   import Qt
from PyQt4.QtGui    import QDialog, QTableWidgetItem

import tank
from tank.common    import TankNotFound

class ReviewItem( QTableWidgetItem ):
    def __init__( self, baseData, review ):
        # extract the important information
        asset               = review.get_asset()
        labels              = asset.get_labels()
        label_dict          = dict([(label.get_entity_type().get_name(),label.get_name()) for label in labels])
        label_dict['name']  = review.get_name()
        
        # initialize the review item
        super(ReviewItem,self).__init__( '%(SceneShot)s %(Department)s (%(ReviewType)s rev:%(name)s)' % label_dict )
        self.setCheckState( Qt.Unchecked )
        
        # store the review instance
        self._baseData      = baseData
        self._reviewLabel   = review.get_name()
        if ( label_dict['ReviewType'] == 'technical' ):
            self._reviewLabel   += ' T'
            
        self._review            = review
    
    def baseData( self ):
        return self._baseData
    
    def reviewData( self ):
        return '%s|%s' % (self._reviewLabel,self._review)
    
    def review( self ):
        return self._review

class ReviewMapperDialog( QDialog ):
    def __init__( self, parent = None ):
        super(ReviewMapperDialog,self).__init__(parent)
        
        # load the ui
        uifile = os.path.join( os.path.dirname(__file__),'resource',os.path.splitext(os.path.basename(__file__))[0] + '.ui' )
        PyQt4.uic.loadUi( uifile, self )
        if ( parent ):
            self.setPalette(parent.tree.palette())
        
        # create connections
        self.uiDialogBTNS.accepted.connect( self.accept )
        self.uiDialogBTNS.rejected.connect( self.reject )
    
    def findAncestor( self, container_type, tank_obj, depth = 0, maxDepth = 10 ):
        # return this object if its type matches the container type
        if ( tank_obj.get_entity_type().get_name() == container_type ):
            return tank_obj
        
        # if we've hit the max recusion depth then quit
        if ( depth == maxDepth ):
            return None
        
        # otherwise, try to look through the dependencies for the container type
        deps = tank_obj.dependencies.get_value()
        if ( deps ):
            for dep in deps:
                result = self.findAncestor( container_type, dep, depth + 1, maxDepth )
                if ( result ):
                    return result
        
        return None
    
    def collectReviews( self, tank_obj ):
        # extract the tank information for the reviews
        rev_type            = tank_obj.get_entity_type()
        container           = tank_obj.get_container()
        container_address   = str(container)
        
        # replace the review type for the container
        if ( 'ReviewType(creative)' in container_address ):
            container_address = container_address.replace('ReviewType(creative)','ReviewType(technical)')
        elif ( 'ReviewType(technical)' in container_address ):
            container_address = container_address.replace('ReviewType(technical)','ReviewType(creative)')
        else:
            return []
        
        # create the mapping
        try:
            matching            = tank.find(container_address).get_object()
        except TankNotFound:
            return []
        
        # make sure our revisions are of the same type, and come from the same root maya scene
        mscene    = self.findAncestor( 'MayaScene', tank_obj )
        revisions = [ rev for rev in matching.get_revisions() if rev.get_entity_type() == rev_type and self.findAncestor( 'MayaScene', rev ) == mscene ]
        revisions.reverse()
        
        return revisions
    
    def currentReviews( self ):
        """
            :remarks        return the currently checked review items from
                            the interface
            :return         <dict> { <str> tank_address: <list> [ <str> tank_address, .. ], }
        """
        output = {}
        
        # loop through all the items looking for the checked ones
        for row in range(self.uiMapperTABLE.rowCount()):
            for col in range(self.uiMapperTABLE.columnCount()):
                item = self.uiMapperTABLE.item(row,col)
                if ( item.checkState() == Qt.Checked ):
                    addr = item.baseData()
                    output.setdefault(addr,[])
                    output[addr].append( item.reviewData() )
                    
        return output
    
    def setShotData( self, shotData ):
        """
            :remarks        sets the table with the inputed reviews and their related
                            reviews based on if the input is creative or technical
                            
            :param          reviews
            :type           <list> [ <tank.Object>, .. ]
        """
        # block the signals & ui updates
        self.uiMapperTABLE.setUpdatesEnabled(False)
        self.uiMapperTABLE.blockSignals(True)
        
        # clear the table
        while ( self.uiMapperTABLE.rowCount() ):
            self.uiMapperTABLE.removeRow(0)
        
        # create the columns
        self.uiMapperTABLE.setColumnCount(len(shotData))
        self.uiMapperTABLE.setHorizontalHeaderLabels( [ '%(shot)s %(department)s (%(review_type)s rev: %(rev_name)s)' % shot['aux_data'] for shot in shotData ] )
        
        # update the resize policy
        header = self.uiMapperTABLE.horizontalHeader()
        for c in range( self.uiMapperTABLE.columnCount() ):
            header.setResizeMode(c,header.ResizeToContents)
        
        # create the row contents
        for col, shot in enumerate(shotData):
            shot_label  = shot['aux_data']['rev_name']
            shot_obj    = shot['aux_data']['tank_asset_rev_obj'].get_object()
            reviews     = self.collectReviews(shot_obj)
            
            for row, review in enumerate(reviews):
                # create the new row if necessary
                if ( row == self.uiMapperTABLE.rowCount() ):
                    self.uiMapperTABLE.insertRow(row)
                
                # create the review item
                item = ReviewItem('%s|%s' % (shot_label,shot_obj),review)
                
                # by default, check the first item in the list
                if ( not row ):
                    item.setCheckState(Qt.Checked)
                
                # add the item to the table
                self.uiMapperTABLE.setItem( row, col, item )
        
        # restore the signals & ui updates
        self.uiMapperTABLE.setUpdatesEnabled(True)
        self.uiMapperTABLE.blockSignals(False)

    # define static methods
    @staticmethod
    def collectPairing( shotData, parent = None ):
        dlg = ReviewMapperDialog( parent )
        dlg.setShotData(shotData)
        if ( dlg.exec_() ):
            return (dlg.currentReviews(),True)
        return ([],False)

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

