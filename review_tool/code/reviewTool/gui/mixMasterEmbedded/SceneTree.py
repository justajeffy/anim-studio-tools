import sys, os
import subprocess
import time
from PyQt4 import QtCore, QtGui, uic

import tank

from __init__ import get_image_thumbnail
import resources

from  rv_tools.util import get_frame_sequence_source

CELL_IMAGE_PATH_ROLE = 99
CELL_PREV_DRAWN_PATH_ROLE = 98


#######################################
## global variables
#######################################
import time
class PandoraBox:
    def __init__(self):
        self._itemHash = {}
        self._itemIsLocked = {}

    def has_item(self, key):
        return self._itemHash.has_key(key)

    def create_key(self, key):
        self._itemIsLocked[key] = False

    def lock_item(self, key):
        """
        """
        if not(self._itemIsLocked.has_key(key)):
            # lock doesn't exist, created lock and then lock it immediately
            self._itemIsLocked[key] = True

        else:
            # lock does exist, wait for it
            while (self._itemIsLocked[key]):
                time.sleep(0.01)

            self._itemIsLocked[key] = True

    def free_item(self, key):
        self._itemIsLocked[key] = False

    def has_item(self, key):
        return self._itemHash.has_key(key)

    def get_item(self, key):
        return self._itemHash[key]

    def set_item(self, key, value):
        self._itemHash[key] = value



thumbPandora = PandoraBox()

class FrameThumbnailDelegate(QtGui.QItemDelegate):
    '''
    The class displays the frame image in table cell
    '''
    FRAME_TEXT_WIDTH = 45  # roughly the width of diplaying 4 digits in size 10 font.

    def __init__(self, parent = None):
        '''
        The class initialize with the parent table view class,
        This is essential because all thumb cache gets stored in parent,
        the delegate class does not store data.
        The path to the image that needs to be drawn comes from the model data.
        '''
        QtGui.QItemDelegate.__init__(self, parent)
    
    def paint (self, painter, option, index):
        '''
        Paint the image as well as the frame index in the table cell
        '''
        QtGui.QItemDelegate.drawBackground( self, painter, option, index );

        # draw the text
        self._paintHelper_text(painter, option, index)

        # get the path to the image
        path = str(index.data(CELL_IMAGE_PATH_ROLE).toString())

        # now get the item and paint it
#        global thumbPandora
        thumbPandora.lock_item(path)

        # draw the item
        if thumbPandora.has_item(path):
            self.parent().model().setData(
                                     index,
                                     QtCore.QVariant(path),
                                     CELL_PREV_DRAWN_PATH_ROLE
                                     )
            prevPath = str(index.data(CELL_PREV_DRAWN_PATH_ROLE).toString()).strip()
            self._paintHelper_image(painter, option, index, thumbPandora.get_item(path))


        else:

            prevPath = str(index.data(CELL_PREV_DRAWN_PATH_ROLE).toString()).strip()
            if prevPath!="":
                self._paintHelper_image(painter, option, index, thumbPandora.get_item(prevPath))

        thumbPandora.free_item(path)


    def _paintHelper_text(self, painter, option, index):
        if FrameThumbnailDelegate.FRAME_TEXT_WIDTH == 0 : return
        painter.save()

        x1, y1, x2, y2 = option.rect.getCoords()
        h = y2 - y1
        w = x2 - x1 - FrameThumbnailDelegate.FRAME_TEXT_WIDTH

        pal = QtGui.QApplication.palette()
        ############## draw text

        if (option.state & QtGui.QStyle.State_Selected):
            painter.setBrush(  pal.brush(QtGui.QPalette.Active, QtGui.QPalette.HighlightedText) )
            painter.setPen(    QtGui.QPen(pal.color(QtGui.QPalette.Active, QtGui.QPalette.HighlightedText)) )
        else:
            painter.setPen(     QtGui.QPen( pal.color(QtGui.QPalette.Active, QtGui.QPalette.Text)) )
            painter.setBrush(   QtGui.QBrush( pal.color(QtGui.QPalette.Active, QtGui.QPalette.Text)) )


        painter.drawText(
                            x1+5, y1+h/2+5,
                            index.data(QtCore.Qt.DisplayRole).toString()
                          )
        painter.restore()

    def _paintHelper_image(self, painter, option, index, imgData):
        painter.save()

        # get the size of the cell
        x1, y1, x2, y2 = option.rect.getCoords()
        h = y2 - y1
        w = x2 - x1 - FrameThumbnailDelegate.FRAME_TEXT_WIDTH

        ############## draw image
        if imgData != None:
            try:
                pix = QtGui.QPixmap(imgData)

                pixAlpha = QtGui.QPixmap(pix.width(), pix.height())

                if (option.state & QtGui.QStyle.State_Selected):
                    pixAlpha.fill(QtGui.QColor(255,255,255))
                else:
                    pixAlpha.fill(QtGui.QColor(190,190,190))

                pix.setAlphaChannel(pixAlpha)

                # get the size of the image
                ix1, iy1, ix2, iy2 = pix.rect().getCoords()
                ih = iy2-iy1
                iw = ix2-ix1

                # do some cropping to ensure the image fits in cell
                pix2 = pix.copy(max(0,(iw-w)/2), max(0, (ih-h)/2), min(w, iw), min(h, ih))

                # bounds describe the size/position in cell to draw image
                bounds = QtCore.QRect(pix2.rect())
                bounds.moveCenter(option.rect.center())
                bounds.moveRight(option.rect.right())

                painter.drawPixmap(bounds, pix2)
            except:
                import traceback
                traceback.print_exc()

        painter.restore()

    def setModelData(self, editor, model, index):
        QtGui.QItemDelegate.setModelData(self, editor, model, index)

    def createEditor(self, parent, option, index):
        return None




class DummyScrubber(QtGui.QWidget):
    '''
    Dummy Scrubber exist because the Qt Table view doesn't can release the mouse once it's grabed.
    Hence the grab mouse functionality is deligated to this class.
    This class handles the mouse move and manipulate parent table view data
    '''
    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        self.setMaximumHeight(0)
        self.setMaximumWidth(0)
        self.sourceData = None
        self.pressedModelIndex = None


    def scrubFrame(self, modelIndex):
        '''
        The original frame is stored, and the mouse is grabbed.
        The original mouse position is also capture for offset calculation.
        This function is involved when by parent table when a cell is clicked.
        '''
        global stateScrubbing
        stateScrubbing = True
        self.startScrubX = None

        self.pressedModelIndex = modelIndex

        self.pressedModelIndex_parent = None

        if modelIndex.parent().row() == -1:
            self.sourceData = self.parent().getShotData(modelIndex.row())
        else:
            self.pressedModelIndex_parent = modelIndex.parent()
            self.sourceData = self.parent().getShotData(  index = self.pressedModelIndex_parent.row(),
                                                        childIndex = modelIndex.row())

        key=self.parent().headerOrder[ modelIndex.column() ] # Start or End
        self.oriFrame = self.sourceData[key]
        self.oriFrame = max(self.oriFrame, self.sourceData["Min"])
        self.oriFrame = min(self.oriFrame, self.sourceData["Max"])

        self.grabMouse(QtCore.Qt.SizeHorCursor)

    def mouseMoveEvent(self, event):
        '''
        Use the mouse offset to relatively calculate the frame offset.
        '''
        if self.startScrubX==None:
            self.startScrubX = event.x()

        mouseOffset = event.x() - self.startScrubX
        frameOffset = mouseOffset / 21

        key=self.parent().headerOrder[ self.pressedModelIndex.column() ] # Start of End

        offsetFrame = self.oriFrame + frameOffset
        offsetFrame = max(offsetFrame, self.sourceData["Min"])
        offsetFrame = min(offsetFrame, self.sourceData["Max"])

        self.sourceData[key] = offsetFrame

        if self.pressedModelIndex_parent==None:
            self.parent().setShotData(self.sourceData, self.pressedModelIndex.row())
        else:
            self.parent().setShotData(    self.sourceData,
                                        index = self.pressedModelIndex_parent.row(),
                                        childIndex = self.pressedModelIndex.row()
                                        )


        self.parent().syncScrubData(self.pressedModelIndex, self.sourceData)

        self.parent().update(self.pressedModelIndex)

    def mouseReleaseEvent(self, event):
        '''
        Release the mouse, the whole reason for this class is because the parent table view
        can't release the mouse properly.
        '''
        self.releaseMouse()

        global stateScrubbing

        stateScrubbing = False

        self.parent().signalScrubFinish(self.pressedModelIndex, self.sourceData)

        QtGui.QApplication.restoreOverrideCursor()



class SceneTree(QtGui.QTreeView):
    '''
    The scene table class list the source data in mix master session.
    Currently the table model stores the latest data source
    '''
    def __init__(self, parent, thumbTmpRoot):
        QtGui.QTreeView.__init__(self, parent)
        
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.setTabKeyNavigation(False)
        self.thumbTmpRoot = thumbTmpRoot
        self.setSortingEnabled(True)
        
        self._sourceTriggerSignalBlocked = False

        self.headerOrder = [
                            "Source Path",
                            "L - Source",
                            "R - Source",
                            "Audio",
                            "Audio Override",
                            "Audio Offset",
                            "Source Name",
                            "Scene",
                            "Shot",
                            "Start",
                            "End",
                            "Audio Start",
                            "Duration",
                            "Min",
                            "Max",
                            "Head",
                            "Tail"
                            ]

        self.headerSizeHash = {
                               "Source Path":-50,  # take up the rest of the space
                               "Source Name":-99,
                               "L - Source": 20,
                               "R - Source": 20,
                               "Audio":0,
                               "Audio Override":20,
                               "Audio Offset":0,
                               "Audio Start": -1,
                               "Scene":0,
                               "Shot":0,
                               "Start":125,
                               "End":125,
                               "Duration":60,
                               "Revision":40,
                               "Max":0,
                               "Min":0,
                               "Head":40,
                               "Tail":40
                               }

        self.headerDataType = {
                               "Source Path":str,
                               "Source Name":str,
                               "L - Source":str,
                               "R - Source":str,
                               "Audio":str,
                               "Audio Override":str,
                               "Audio Offset":int,
                               "Audio Start": int,
                               "Scene":str,
                               "Shot":int,
                               "Start":int,
                               "End":int,
                               "Revision":int,
                               "Max":int,
                               "Min":int,
                               "Head":int,
                               "Tail":int
                               }


        self.delegateHash = {
                                "Start":FrameThumbnailDelegate,
                                "End":FrameThumbnailDelegate
                             }
        
        self._editableColumns = []
        
        # create the dummy scrubber, used to enable scrubbing
        # see the class for more information for why this is needed.
        self.dummyTracker = DummyScrubber(self)
        self.setMouseTracking(True)
        self.stateRightMouseDown = False
        
        self.connect(self, QtCore.SIGNAL("pressed ( const QModelIndex &)"), self.cellPressed)

    def keyPressEvent( self, event ):
        if event.key()==65 and event.modifiers()==QtCore.Qt.ControlModifier:
            self.emit(QtCore.SIGNAL("select_all_items"))

        elif event.key()==16777223:
            self.emit(QtCore.SIGNAL("remove_selection"))
        
        else:
            QtGui.QTreeView.keyPressEvent(self, event)
    
    def canEditColumn( self, col ):
        return col in self._editableColumns
    
    def closeEditor( self, editor, editHint ):
        # edit the next item by moving left >> right in the row vs up >> down in the column
        if ( editHint == QtGui.QItemDelegate.EditNextItem ):
            # grab the current index
            index       = self.currentIndex()
            
            # close the editor with no hint
            super(SceneTree,self).closeEditor(editor,QtGui.QItemDelegate.NoHint)
            
            # look for the next editable col
            row     = index.row()
            col     = index.column() + 1
            found   = False
            for c in range(col,self.model().columnCount()):
                if ( self.canEditColumn(c) ):
                    col = c
                    found = True
                    break
            
            # if it was not found, increment the row and then start the process again
            if ( not found ):
                row += 1
                for c in range(self.model().columnCount()):
                    if ( self.canEditColumn(c) ):
                        col = c
                        found = True
                        break
            
            if ( found ):
                next = index.sibling( row, col )
                if ( next and next.isValid() ):
                    self.setCurrentIndex(next)
                    self.edit(next)
            
        # edit the previous item by moving left << right in the row vs up << down in the column
        elif ( editHint == QtGui.QItemDelegate.EditPreviousItem ):
            # grab the current index
            index       = self.currentIndex()
            
            # close the editor with no hint
            super(SceneTree,self).closeEditor(editor,QtGui.QItemDelegate.NoHint)
            
            # look for the next editable col
            row     = index.row()
            col     = index.column() - 1
            found   = False
            for c in range(col,-1,-1):
                if ( self.canEditColumn(c) ):
                    col = c
                    found = True
                    break
            
            # if it was not found, increment the row and then start the process again
            if ( not found ):
                row -= 1
                for c in range(self.model().columnCount() - 1, -1, -1):
                    if ( self.canEditColumn(c) ):
                        col = c
                        found = True
                        break
            
            if ( found ):
                prev = index.sibling( row, col )
                if ( prev and prev.isValid() ):
                    self.setCurrentIndex(prev)
                    self.edit(prev)
        else:
            super(SceneTree,self).closeEditor(editor,editHint)

    def clearThumbCache(self, hint=None):
        global thumbCache

        if hint==None:
            thumbCache = {}
        else:
            for k in thumbCache.keys():
                if k.find(hint)!=-1:
                    del(thumbCache[k])


    def cellPressed(self, modelIndex):
        '''
        The table cell is pressed, only scrub if start scrubbing if Start or End cell is selected
        This happens after mousePressEvent
        '''

        if self.stateRightMouseDown and \
            ( modelIndex.column()==self.headerOrder.index("Start") or
              modelIndex.column()==self.headerOrder.index("End")):

            self.dummyTracker.scrubFrame(modelIndex)

        else:
            if modelIndex.parent().row()==-1:
                index = modelIndex.row()
                childIndex = None
            else:
                index = modelIndex.parent().row()
                childIndex = modelIndex.row()

            self.emit(QtCore.SIGNAL("sourceCellClicked"),
                      { "index": index,
                        "childIndex":childIndex,
                        "columnIndex":modelIndex.column(),
                        "columnHeader":self.headerOrder[modelIndex.column()]
                        }
                      )

    def updateHeaderSize( self, index, oldSize, newSize ):
        # make sure the tree is visible and has items, otherwise ignore the change
        if ( not (self.isVisible() and self.model().rowCount()) ):
            return
            
        if ( not (0 <= index and index < len(self.headerOrder)) ):
            print 'column index is out of range:', index
        
        # set the column width in the size hash
        columnName  = self.headerOrder[index]
        self.headerSizeHash[columnName] = newSize
        
    def mousePressEvent(self, event):
        self.stateRightMouseDown = (event.button()==QtCore.Qt.RightButton)

        QtGui.QTreeView.mousePressEvent(self, event)
        self.emit(QtCore.SIGNAL("sourceSelectionChanged"))

    def mouseReleaseEvent(self, event):
        QtGui.QTreeView.mouseReleaseEvent(self, event)
        self.stateRightMouseDown = False
        self.emit(QtCore.SIGNAL("sourceSelectionChanged"))


    def mouseDoubleClickEvent(self, event):
        QtGui.QTreeView.mouseDoubleClickEvent(self, event)
        if ( not self.sourceTriggerSignalBlocked() ):
            self.emit(QtCore.SIGNAL("sourceEditTrigger"))

    def setEditableColumns( self, columns ):
        self._editableColumns = columns

    def setDelegate(self):
        '''
        Set the delegate for column according to the delegate hash.
        '''

        for header in self.headerOrder:
            if self.delegateHash.has_key(header):

                delegateClass = self.delegateHash[header]
                self.setItemDelegateForColumn(
                                          self.headerOrder.index(header),
                                          delegateClass(self)
                                          )


    def clear(self):
        self.model().clear()

    def clearSelection(self):
        self.selectionModel().clear()

    def getSelectedRow(self):
        smodel = self.selectionModel()

        rowSelected = []

        for eachSelectedIndex in smodel.selectedIndexes():
            if eachSelectedIndex.parent().row()==-1:
                rowIndex = ( eachSelectedIndex.row(), None)
            else:
                rowIndex = ( eachSelectedIndex.parent().row(), eachSelectedIndex.row() )

            if not(rowIndex in rowSelected ):
                rowSelected.append(rowIndex )


        rowSelected.sort()

        return rowSelected


    def selectRows(self, rowIndexList):
        self.clearSelection()
        for i in rowIndexList:
            QtGui.QApplication.processEvents()
            self.selectRow(i)

    def selectAll(self):
        for i in range(self.getRowCount()):
            QtGui.QApplication.processEvents()
            self.selectRow(i)

    def selectRow(self, index, childIndex = None):
        parentItem, rowIndex = self._resolveParentAndRowIndex(index, childIndex)

        for colIndex in range(self.model().columnCount()):
            mIndex = self.model().indexFromItem(parentItem.child(rowIndex, colIndex))
            self.selectionModel().select( mIndex, QtGui.QItemSelectionModel.Select)


    def _resolveParentAndRowIndex(self, index, childIndex=None):

        if type(index)==tuple or type(index)==list or type(index)==set:
            index, childIndex =  index

        if childIndex==None:
            parentItem = self.model().invisibleRootItem()
            rowIndex = index

        else:
            parentItem = self.model().itemFromIndex( self.model().index(index, 0) )
            rowIndex = childIndex

        return parentItem, rowIndex


    def insertEmptyRow(self, index, childIndex=None, flgAutoExpand=True):
        parentItem, rowIndex = self._resolveParentAndRowIndex(index, childIndex)
        parentItem.insertRow( rowIndex, [QtGui.QStandardItem () for i in range(len(self.headerOrder))] )

        # expand the parent to see the children upon add
        if flgAutoExpand:
            # ensure the parent is expanded
            mIndex = self.model().indexFromItem( parentItem )
            self.expand(mIndex)

            # scroll to the child
            cIndex = self.model().indexFromItem(
                                                parentItem.child(rowIndex, 0)
                                                )
            self.scrollTo(cIndex)



    def removeShotRow(self, index, childIndex=None):
        parentItem, rowIndex = self._resolveParentAndRowIndex(index, childIndex)

        parentItem.removeRow( rowIndex)


    def addShot(self, shotDataHash, index=None, childIndex=None, flgAutoExpand=True):
        '''
        Add a brand new source to mix master.
        The index are for the rows
        '''
        if index==None:
            index = self.model().rowCount()

        if not shotDataHash.has_key('Min'):
            not_used, not_used, shotDataHash['Min'], shotDataHash['Max'] = get_frame_sequence_source(
                                                os.path.join(shotDataHash['Source Path'],shotDataHash['Source Name']))

        self.insertEmptyRow(index, childIndex, flgAutoExpand)

        # now set the data to the row
        self.setShotData(shotDataHash, index, childIndex=childIndex )


    def setShotData(self, shotDataHash, index, childIndex=None, syncAllChild=False):
        '''
        Update the data source, create the thread to create the thumbnails
        '''
        import time
        stime = time.time()
        global thumbPandora
        parentItem, rowIndex = self._resolveParentAndRowIndex(index, childIndex)

        for header in shotDataHash.keys():
            if not header in self.headerOrder: continue

            item = parentItem.child(rowIndex, self.headerOrder.index(header))
            item.setData( QtCore.QVariant( shotDataHash[header] ), QtCore.Qt.DisplayRole )

            if self.delegateHash.has_key(header) and self.delegateHash[header]== FrameThumbnailDelegate:#:"Start" or header=="End":
                path = os.path.join(shotDataHash["Source Path"], os.path.join(shotDataHash["Source Name"]))
                path = path.replace( "#", "%04d" % shotDataHash[header] )

                item.setData( QtCore.QVariant(path),
                              CELL_IMAGE_PATH_ROLE
                              )

                thumbPandora.create_key(path)

                if False:
                    t = ThumbGenerateThread(self, path, self.headerSizeHash[header] - FrameThumbnailDelegate.FRAME_TEXT_WIDTH, self.thumbTmpRoot)
                    t.start()
                else:
                    thumbPandora.lock_item(path)

                    width = self.headerSizeHash[header] - FrameThumbnailDelegate.FRAME_TEXT_WIDTH
                    if not(thumbPandora.has_item(path)):
                        img_path = get_image_thumbnail(path, width, width, self.thumbTmpRoot)
                        thumbPandora.set_item(path, img_path)

                    thumbPandora.free_item(path)

        # set the data of the child for the column to be the same data
        if syncAllChild and parentItem.row()==-1:
            for i in range(self.getRowCount(rowIndex)):
                self.setShotData(shotDataHash, rowIndex, childIndex=i)

        item = item = parentItem.child(rowIndex, 0)
        item.setSizeHint(QtCore.QSize(100,30))


    def getRowCount(self, index=None):
        if index==None:
            return self.model().rowCount()
        else:
            return self.model().invisibleRootItem().child(index, 0).rowCount()



    def getShotAndVariantData(self, index, flgAsList=False):
        # first gather the parent data
        if flgAsList:
            varData = []
            varData.append(self.getShotData(index = index))
        else:
            varData = {}
            varData[(index,  None)] = self.getShotData(index = index)

        # now gather the child data
        for childIndex in range(self.getRowCount(index)):
            if flgAsList:
                varData.append( self.getShotData(index = index, childIndex = childIndex) )
            else:
                varData[ (index, childIndex) ] = self.getShotData(index = index, childIndex = childIndex)

        return varData


    def getVariantShotData(self, rowIndex):
        varData = []
        for i in range(self.getRowCount(rowIndex)):
            varData.append(
                           self.getShotData(index = rowIndex, childIndex = i)
                           )

        return varData


    def findShotData(self, rowIndex, searchCriteria={}):
        allShotData = self.getShotAndVariantData(rowIndex)

        shotResultSet = []

        for i in allShotData:
            shotData = allShotData[i]

            flgMatch=True
            for searchKey in searchCriteria:

                searchValue = searchCriteria[searchKey]

                if shotData[searchKey]!= searchValue:

                    flgMatch=False

            if flgMatch:
                shotResultSet.append(shotData)

        return shotResultSet

    def getShotData(self, index, childIndex=None):
        '''
        Returns the source data from the row.
        '''
        parentItem, rowIndex = self._resolveParentAndRowIndex(index, childIndex)

        data = {}
        for header in self.headerOrder:
            data[header] =  parentItem.child(   rowIndex,
                                                self.headerOrder.index(header) ).data(QtCore.Qt.DisplayRole)

            if header in self.headerDataType:
                if self.headerDataType[header]==int:
                    if data[header].toString()!='':
                        raw = data[header].toString()
                        try:
                            data[header] = int(raw)
                        except:
                            print "Warning: failed to cast data to int: %s" % raw
                    else:
                        data[header] = None
                elif self.headerDataType[header]==str:
                    data[header] = str(data[header].toString())

        return data


    def syncScrubData(self, modelIndex, scrubData):
        self.emit(QtCore.SIGNAL("frameScrubChange"), modelIndex, scrubData)


    def signalScrubFinish(self, modelIndex, scrubData):
        self.emit(QtCore.SIGNAL("frameScrubFinish"), modelIndex, scrubData)


    def setupData(self):
        # create the model for this tree
        model = QtGui.QStandardItemModel(0, len(self.headerOrder))
        model.setSupportedDragActions(QtCore.Qt.CopyAction)
        self.setModel(model)
        
        # initialize the header
        header          = self.header()
        header.setStretchLastSection(False)
        
        label_dict      = self.__dict__.get('headerLabel',{})
        icon_dict       = self.__dict__.get('headerIcon',{})
        tooltip_dict    = self.__dict__.get('headerToolTip',{})
        
        for i, key in enumerate(self.headerOrder):
            # set the header item data
            model.setHeaderData( i, QtCore.Qt.Horizontal, QtCore.QVariant(label_dict.get(key,key)) )
            model.setHeaderData( i, QtCore.Qt.Horizontal, QtCore.QVariant(icon_dict.get(key,key)),      QtCore.Qt.DecorationRole )
            model.setHeaderData( i, QtCore.Qt.Horizontal, QtCore.QVariant(tooltip_dict.get(key,key)),   QtCore.Qt.ToolTipRole )
            
            # set the resize mode
            w = self.headerSizeHash.get(key,0)
            
            # update the resizing mode based on the size hint information
            if ( w == -50 ):
                header.setResizeMode( i, header.Stretch )
            elif ( w == 0 ):
                header.hideSection(i)
            else:
                model.setHeaderData(i, QtCore.Qt.Horizontal, QtCore.QVariant(QtCore.QSize(w,0)), QtCore.Qt.SizeHintRole )
        
        self.setDelegate()
    
    def setSourceTriggerSignalBlocked( self, state = True ):
        self._sourceTriggerSignalBlocked = state
    
    def sourceTriggerSignalBlocked( self ):
        return self._sourceTriggerSignalBlocked

if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)

    test = SceneTree(None, '/tmp/mixMaster')
    test.setupData()

    test.setFixedWidth(1000)
    test.setFixedHeight(600)
    test.show()

    test.addShot({'Start': 1050, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/020/sc_100a_020_previs_v721.7283_53795.dir', 'End': 1051, 'Source Name': 'sc_100a_020_previs_v721.#.png'})
    test.setShotData({'Start': 1055, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/020/sc_100a_020_previs_v721.7283_53795.dir', 'End': 1051, 'Source Name': 'sc_100a_020_previs_v721.#.png'},
                     0

                     )
#
#    test.addShot({'Start': 6582, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/020/sc_100a_020_previs_v475.7283_44072.dir', 'End': 6588, 'Source Name': 'sc_100a_020_previs_v475.#.png'})
#    test.addShot({'Start': 1, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/030', 'End': 50, 'Source Name': 'sc_100a_030_previs_v494.7285_46395.mov'})
#    test.addShot({'Start': 6582, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/020/sc_100a_020_previs_v475.7283_44072.dir', 'End': 6588, 'Source Name': 'sc_100a_020_previs_v475.#.png'})
#    test.addShot({'Start': 1, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/030', 'End': 50, 'Source Name': 'sc_100a_030_previs_v494.7285_46395.mov'})
#    test.addShot({'Start': 6647, 'Source Path': '/drd/jobs/hf2/story/sc_100a_rndtestWillgoAway/030/output/sc_100a_030_previs_v476', 'End': 6700, 'Source Name': 'sc_100a_030_previs_v476.#.png'})
#    test.addShot({'Start': 2, 'Source Path': '/drd/jobs/hf2/story/sc_100a_rndtestWillgoAway/010/output/sc_100a_010_previs_v510', 'End': 23, 'Source Name': 'sc_100a_010_previs_v510.#.png'})
#
#    test.addShot({'Start': 1085, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/030/sc_100a_030_previs_v723.7285_53803.dir', 'End': 1191, 'Source Name': 'sc_100a_030_previs_v723.#.png'})
#    test.addShot({'Start': 5, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/010/sc_100a_010_previs_v608.7281_52661.dir', 'End': 19, 'Source Name': 'sc_100a_010_previs_v608.#.png'})
#    test.addShot({'Start': 20, 'Source Path': '/tmp/sc_100a_020_previs_v612', 'End': 29, 'Source Name' : 'sc_100a_020_previs_v612.#.png'})
#    test.addShot({'Start': 30, 'Source Path': '/tmp/sc_100a_030_previs_v612', 'End': 124, 'Source Name': 'sc_100a_030_previs_v612.#.png'})
#    test.addShot({'Start': 1192, 'Source Path': '/drd/jobs/hf2/tank/classic_v1/story/sc_100a/040/sc_100a_040_previs_v722.8623_53799.dir', 'End': 1229, 'Source Name': 'sc_100a_040_previs_v722.#.png'})


    sys.exit(app.exec_())




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

