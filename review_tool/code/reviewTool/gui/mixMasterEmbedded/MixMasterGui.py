import os, sys, traceback
import subprocess

from PyQt4 import QtCore, QtGui, uic
import tank

import drGadgets.lib.qtGuiPreference as guiPrefSaver

import resources
from SceneTree import SceneTree

from rv_tools.util import get_frame_sequence_source

Ui_MixMasterMain = uic.loadUiType(resources.fetch("MixMasterGui.ui"))[0]

class MixMasterGui(Ui_MixMasterMain, QtGui.QWidget):
    '''
    The main panel for mix master
    '''
    def __init__(self, parent, prefSaver, thumbTmpRoot):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.tableView = SceneTree(self, thumbTmpRoot = thumbTmpRoot)
        self.tableView.setupData()
        self.tableView.setFrameShape(QtGui.QFrame.NoFrame)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.tableView)
        vbox.setMargin(0)
        self.playListContainer.setLayout(vbox)

        # set the icons
        self.cmdAddSource.setIcon(QtGui.QIcon(resources.fetch("add.png")))
        self.cmdAddSource.setText("")
        self.cmdAddVariantSource.setIcon(QtGui.QIcon(resources.fetch("subadd.png")))
        self.cmdAddVariantSource.setText("")
        self.cmdRemoveSource.setIcon(QtGui.QIcon(resources.fetch("remove.png")))
        self.cmdRemoveSource.setText("")
        self.cmdClearPlaylist.setIcon(QtGui.QIcon(resources.fetch("clearPage.png")))
        self.cmdClearPlaylist.setText("")
        self.cmdMoveSourceUp.setIcon(QtGui.QIcon(resources.fetch("up.png")))
        self.cmdMoveSourceUp.setText("")
        self.cmdMoveSourceDown.setIcon(QtGui.QIcon(resources.fetch("down.png")))
        self.cmdMoveSourceDown.setText("")
        self.cmdEditSource.setIcon(QtGui.QIcon(resources.fetch("pencil.png")))
        self.cmdEditSource.setText("")
        self.cmdSetDefaultPad.setIcon(QtGui.QIcon(resources.fetch("apply.png")))
        self.cmdSetDefaultPad.setText("")
        self.cmdPublish.setIcon(QtGui.QIcon(resources.fetch("publish.png")))
        self.cmdApplyPad.setIcon(QtGui.QIcon(resources.fetch("apply.png")))
        self.cmdApplyPad.setText("")
        self.cmdHelp.setIcon(QtGui.QIcon(resources.fetch("help.png")))
        self.cmdHelp.setText("")
        self.cmdSavePlaylist.setIcon(QtGui.QIcon(resources.fetch("saveFile.png")))
        self.lblMixMasterIcon.setPixmap(QtGui.QPixmap(resources.fetch("tapeRecorder32.png")))
        self.cmdPlay.setIcon(QtGui.QIcon(resources.fetch("play.png")))
        self.cmdImportPlaylist.setIcon(QtGui.QIcon(resources.fetch("page_lightning.png")))
        self.cmdCloseEditSourcePanel.setIcon(QtGui.QIcon(resources.fetch("close.png")))

        self.cmdAddVariantSource.setEnabled(False)

        self.cmdImportPlaylist.setFocus()
        # show the edit source by default
        self.pnlEditSource.hide()
        self.clearSourceEditPanel()

        version = os.environ["MIXMASTER_VERSION"] if os.environ.has_key("MIXMASTER_VERSION") else ""
        self.titleMixMaster = "Mix Master - %s : " % version
        self.setWindowTitleFile("")

        self.connect (  self.tableView,
                QtCore.SIGNAL("frameScrubChange"),
                self.syncFrameScrubChange)

        self.connect (  self.tableView,
                QtCore.SIGNAL("frameScrubFinish"),
                self.handleFrameScrubFinish)

        self.connect (  self.cmdEditAudioPrompt,
                QtCore.SIGNAL("clicked()"),
                self.handleEditAudioPrompt)

        self.connect (  self.cmdClearPlaylist,
                QtCore.SIGNAL("clicked()"),
                self.clearPlaylist)

        self.connect (  self.cmdAddSource,
                QtCore.SIGNAL("clicked()"),
                self.handleAddSource)

        self.connect (  self.cmdRemoveSource,
                QtCore.SIGNAL("clicked()"),
                self.removeSource)

        self.connect (  self.cmdMoveSourceUp,
                QtCore.SIGNAL("clicked()"),
                self.moveSourceUp)

        self.connect (  self.cmdMoveSourceDown,
                QtCore.SIGNAL("clicked()"),
                self.moveSourceDown)

        self.connect ( self.cmdEditSource,
                QtCore.SIGNAL("clicked()"),
                self.editSource
                       )

        self.connect ( self.cmdCloseEditSourcePanel,
                QtCore.SIGNAL("clicked()"),
                self.closeEditSource
                       )

        self.connect ( self.cmdEditSourcePrompt,
                QtCore.SIGNAL("clicked()"),
                self.handleEditSourcePrompt
                       )

        self.connect ( self.cmdAudioSource,
                QtCore.SIGNAL("clicked()"),
                self.handleEditAudioSource
                       )

        self.connect ( self.tableView,
                QtCore.SIGNAL("sourceSelectionChanged"),
                self.handleShotSelectionChanged
                       )

        self.connect ( self.tableView,
                QtCore.SIGNAL("sourceCellClicked"),
                self.handleCellSelectionChanged
                       )

        self.connect ( self.tableView,
                QtCore.SIGNAL("sourceEditTrigger"),
                self.playlistEditTrigger
                       )

        self.connect ( self.spnStart,
                QtCore.SIGNAL("valueChanged (int)"),
                self.syncSourceEditPanelToSelection
                       )

        self.connect ( self.spnEnd,
                QtCore.SIGNAL("valueChanged (int)"),
                self.syncSourceEditPanelToSelection
                       )

        self.connect ( self.spnPerShotAudioOffset,
                QtCore.SIGNAL("valueChanged (int)"),
                self.syncSourceEditPanelToSelection
                       )

        self.connect ( self.spnHeadHold,
                QtCore.SIGNAL("valueChanged (int)"),
                self.syncSourceEditPanelToSelection
                       )

        self.connect ( self.spnTailHold,
                QtCore.SIGNAL("valueChanged (int)"),
                self.syncSourceEditPanelToSelection
                       )

        self.connect ( self.cmdApplyPad,
                QtCore.SIGNAL("clicked()"),
                self.applyFramePadding
                       )

        self.connect ( self.cmdSetDefaultPad,
                QtCore.SIGNAL("clicked()"),
                self.setDefaultPad
                       )

        self.connect ( self.cmdHelp,
                       QtCore.SIGNAL("clicked()"),
                       self.showHelp
                       )

        # state data
        self.stateTankVersionChanging = False
        self.stateSyncing = False
        self.cmdPublish.hide()

        # create the preference saver
        if prefSaver == None:
            self.pref = guiPrefSaver.QtGuiPreference("/tmp/mixMaster.conf")
        else:
            self.pref = prefSaver

        # save the text
        self.pref.bootstrapWidget (  widget      = self.lblDefaultPad,
                                     widgetName  = "defaultFramePadding",
                                     widgetType  =  guiPrefSaver.TEXTBOX
                                     )

        self.pref.bootstrapWidget (  widget      = self.chkUseFrameNum,
                                     widgetName  = "repectFrameNumber",
                                     widgetType  =  guiPrefSaver.CHECKBOX
                                     )

        # save the dialog position and size
        self.pref.bootstrapWidget (  widget      = self,
                                     widgetName  = "dialog window",
                                     widgetType  = guiPrefSaver.WINDOW,
                                     defaultValue = "119;76;1250;921"
                                     )

        # when dialog close, call the save preference function
        self.connect(self,
                     QtCore.SIGNAL("finished (int)"),
                     self.savePref
                     )

    def enablePublishMode(self):
        self.cmdPublish.show()

    def setWindowTitleFile(self, fileName):
        parent = self.parent()
        if parent:
            parent.setWindowIcon(QtGui.QIcon(resources.fetch("tapeRecorder48.png")))
            parent.setWindowTitle(self.titleMixMaster + fileName)
        else:
            self.setWindowIcon(QtGui.QIcon(resources.fetch("tapeRecorder48.png")))
            self.setWindowTitle(self.titleMixMaster + fileName)

    def showHelp(self):
        import subprocess
        subprocess.Popen(['konqueror', 'http://prodwiki.drd.int/mediawiki/index.php/RnD:HF2ProductionToolsList:Previs:UserHelp'])


    def closeEvent(self, event):
        self.pref.save()
        self.tableView.clearThumbCache()
        QtGui.QWidget.closeEvent(self, event)


    def savePref(self):
        self.pref.save()


    def setDefaultPad(self):
        self.lblDefaultPad.setText( str( self.spnFramePadding.value() ))


    def applyFramePadding(self):
        selectedRowList = self.tableView.getSelectedRow()
        framepad = int( self.spnFramePadding.value() )

        if len(selectedRowList)>1:
            reply = QtGui.QMessageBox.warning(self, self.tr("Apply Padding"),
                                              "Apply padding of %s frames to all selected source?" % framepad,
                                              self.tr("Ca&ncel"),
                                              self.tr("&Ok"))

            if reply:
                for rowIndex in selectedRowList:
                    shotData = self.tableView.getShotData(rowIndex)

                    frameMax = shotData["Max"]
                    frameMin = shotData["Min"]
                    shotData["Start"] = frameMin + framepad
                    shotData["End"]   = frameMax - framepad

                    if shotData["Start"] <= shotData["End"]:
                        self.tableView.setShotData(shotData, rowIndex)

        else:
            frameMin = int(self.lblFrameMin.text())
            frameMax = int(self.lblFrameMax.text())
            frameStart = frameMin + framepad
            frameEnd = frameMax - framepad

            if frameStart <= frameEnd:
                self.spnStart.setValue(  frameStart  )
                self.spnEnd.setValue( frameEnd  )



    def clearSourceEditPanel(self):

        self.txtEditSourcePath.setText("")

        self.spnStart.setValue(0)
        self.spnEnd.setValue(0)

        self.lblFrameMin.setText("")
        self.lblFrameMax.setText("")


    def syncSceneShotVersionBar(self):
        '''
        update the tank version combo by looking at the path
        populate the tank versions combo box if the source comes from tank
        '''
        if not self.cboSourceVersion.isVisible():
            return

        self.stateTankVersionChanging = True
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # get the source data
        sourceData = self.tableView.getShotData(self.tableView.getSelectedRow()[0])

        if isEditSourceMovie(sourceData["Source Name"]):
            path = os.path.join(sourceData["Source Path"], sourceData["Source Name"])
        else:
            path = sourceData["Source Path"]

        # check if it specify a tank version
        rev = None
        try:
            rev = tank.find(path)
        except tank.common.errors.TankNotFound, e:
            pass

        self.cboSourceVersion.setEnabled(False)
        self.cboSourceVersion.clear()

        if rev:
            if rev.system.has_key('type'):  # tank 1.14
                revType = rev.system.type.system.name
            else:
                revType = rev.system.type_object.get_name()

            if revType in ["DailiesRenderSeq", "DailiesRenderMov", "Movie", "Frames"]:
                self.cboSourceVersion.setEnabled(True)

                revList = filter(lambda x: x[0]==revType, rev.asset.revisions.keys())
                revList = map(lambda x: x[1], revList)

                revList.sort()
                revList.reverse()

            # populate, then select the default version
            self.cboSourceVersion.addItems(revList)
            self.cboSourceVersion.setCurrentIndex(revList.index(rev.system.name))

        self.stateTankVersionChanging = False
        QtGui.QApplication.restoreOverrideCursor()



    def syncFrameScrubChange(self, modelIndex, dataHash):

        self.lblHint.setText(" range: %s-%s     in handle: % 2d     out handle: % 2d     duration: %2d" % (
                                dataHash["Min"],
                                dataHash["Max"],
                                dataHash["Start"] - dataHash["Min"],
                                dataHash["Max"] - dataHash["End"],
                                dataHash["End"] - dataHash["Start"]

                             ))

    def handleFrameScrubFinish(self, modelIndex, dataHash):
        self.stateSyncing = True

        self.spnStart.setValue(dataHash["Start"])
        self.spnEnd.setValue(dataHash["End"])
        self.lblHint.setText("")

        self.tableView.setShotData({'Duration': dataHash["End"] - dataHash["Start"]},
                                        index=modelIndex.row(), syncAllChild=True)

        self.stateSyncing = False

    def sycnSourceEditPanelFrameRange(self, fullPath = None, refreshRange=False):
        '''
        Determine the max and min for the frames in the folder
        Further set the gui, the max and min as well as adjust the frame range to the max and min
        todo: set the frame range with some padding
        '''

        frameMin = self.lblFrameMin.text()
        frameMax = self.lblFrameMax.text()

        if refreshRange:
            pad = int ( self.lblDefaultPad.text() )
            if (int(frameMax) - int(frameMin)) < pad*2:
                pad=0

            self.spnStart.setValue(int(frameMin) + pad)
            self.spnEnd.setValue(int(frameMax) - pad)


    def syncSelectionToSourceEditPanel(self):
        if self.stateSyncing:
            return
        else:
            self.stateSyncing = True

        selectedRowList = self.tableView.getSelectedRow()

        self.grpEditSource.setEnabled(False)
        self.grpEditRange.setEnabled(False)
        self.grpEditFrameHold.setEnabled(False)
        self.cmdRemoveSource.setEnabled(False)
        self.cmdPlaySelected.setEnabled(False)
        self.cmdRemoveSource.setEnabled(False)
        self.clearSourceEditPanel()
        self.txtAudioSource.setText("")
        self.chkAudioSource.setChecked(False)
        self.spnPerShotAudioOffset.setValue(0)

        if len( selectedRowList )>=1:
            self.cmdRemoveSource.setEnabled(True)
            self.cmdRemoveSource.setEnabled(True)
            self.cmdPlaySelected.setEnabled(True)

        if len( selectedRowList )==1:
            self.cmdEditSource.setEnabled(True)
            self.grpEditSource.setEnabled(True)
            self.grpEditRange.setEnabled(True)
            self.grpEditFrameHold.setEnabled(True)


        if not( self.pnlEditSource.isVisible() ):
            self.stateSyncing = False
            return

        if len( selectedRowList )==1:
            # sync data
            dataHash = self.tableView.getShotData(selectedRowList[0])

            self.txtEditSourcePath.setText(
                                           os.path.join(
                                                        dataHash["Source Path"],
                                                        dataHash["Source Name"])
                                           )

            # by default sycnSourceEditPanelFrameRange, use the default range, so
            # override it with thde data
            if dataHash["Start"]!=None: self.spnStart.setValue(dataHash["Start"])
            if dataHash["End"]!=None: self.spnEnd.setValue(dataHash["End"])

            if dataHash["Head"]!=None: self.spnHeadHold.setValue(dataHash["Head"])
            if dataHash["Tail"]!=None: self.spnTailHold.setValue(dataHash["Tail"])

            if dataHash["Max"]!=None: self.lblFrameMax.setText( str(dataHash["Max"]) )
            if dataHash["Min"]!=None: self.lblFrameMin.setText( str(dataHash["Min"]) )

            self.chkAudioSource.setChecked(dataHash["Audio Override"]=="*")
            self.txtAudioSource.setText(dataHash["Audio"])
            self.spnPerShotAudioOffset.setValue(0 if dataHash["Audio Offset"]==None else dataHash["Audio Offset"])


        self.overrideAudioPanel.setEnabled(self.chkAudioSource.isChecked())
        self.stateSyncing = False


    def syncSourceEditPanelToSelection(self, notUsed=None):
        '''
        Sync the data in the edit source panel to the selection
        '''

        if self.stateSyncing:
            return
        else:
            self.stateSyncing = True

        if self.tableView.getSelectedRow() and len(self.tableView.getSelectedRow())==1:
            selectedRow = self.tableView.getSelectedRow()[0]
            dataHash = self.tableView.getShotData( selectedRow )
            dataHash["Start"] = int(self.spnStart.value())
            dataHash["End"] = int(self.spnEnd.value())
            dataHash["Duration"] = dataHash["End"] - dataHash["Start"]

            self.tableView.setShotData(dataHash, selectedRow)

            # for head/tail holding and all variants should be synced, and synced to the top row index
            rootRowIndex = selectedRow[0]

            dataHash={}
            dataHash["Head"] = int(self.spnHeadHold.value())
            dataHash["Tail"] = int(self.spnTailHold.value())
            dataHash["Audio Override"] = "*" if self.chkAudioSource.isChecked() else ""
            dataHash["Audio"] = self.txtAudioSource.text()
            dataHash["Audio Offset"] = self.spnPerShotAudioOffset.value()


            self.tableView.setShotData(dataHash, index=rootRowIndex, syncAllChild=True)

        self.overrideAudioPanel.setEnabled(self.chkAudioSource.isChecked())

        self.stateSyncing = False


    def handleCellSelectionChanged(self, dataHash):
        newData = {}
        newData[dataHash["columnHeader"]] = 0

        if dataHash["columnHeader"]=="L - Source" or dataHash["columnHeader"]=="R - Source":
            selectIndex= (dataHash["index"], dataHash["childIndex"])
            self.handleChooseShotVariant( selectIndex, selectedKey=dataHash["columnHeader"])


    def handleShotSelectionChanged(self):
        rowSelection = self.tableView.getSelectedRow()

        self.cmdAddSource.setEnabled(   len(rowSelection)==0 or
                                        # if int then only has row index, means it's main shots
                                        (len(rowSelection)<=1 and rowSelection[0][1]==None)
                                        )

        self.cmdAddVariantSource.setEnabled(len(rowSelection)==1)

        # check if all of the selected row is a sub row, if so disable shuffle shot
        subRowSelection = filter(lambda row: row[1]!=None, rowSelection)

        self.cmdMoveSourceUp.setEnabled(len(subRowSelection)   ==0)
        self.cmdMoveSourceDown.setEnabled(len(subRowSelection) ==0)

        self.syncSelectionToSourceEditPanel()


    def handleEditAudioSource(self):
        fileName = self.promptEditSource(
                                         default=self.txtAudioSource.text(),
                                         title="Import Audio Source",
                                         filter="Audio (*.wav)"
                                     )

        if fileName!=None:
            self.txtAudioSource.setText(fileName)
            self.syncSourceEditPanelToSelection()

    def handleEditSourcePrompt(self):
        '''
        When the current source is edited
        '''

        from __init__ import SUPPORTED_IMAGE, SUPPORTED_MOVIE
        image_list = " ".join( ["*.%s" % ext for ext in SUPPORTED_IMAGE] )
        movie_list = " ".join( ["*.%s" % ext for ext in SUPPORTED_MOVIE] )

        filter = "image sequence file (%(image_list)s);; movie (%(movie_list)s);;all files (*.*)" % vars()

        fileName = self.promptEditSource(
                                         default = self.txtEditSourcePath.text(),
                                         title="Import Source Sequence - select a frame from sequence",
                                         filter=filter,
                                         flgMultiSelection=False
                                         )

        if fileName!=None:
            fileName = str(fileName)
            result = get_frame_sequence_source(fileName)
            if result!=None:
                filePath, fileName, frameMin, frameMax = result

                shotHash = {}
                shotHash["Source Path"] = filePath
                shotHash["Source Name"] = fileName
                shotHash["Start"]       = int(self.spnStart.value())
                shotHash["End"]         = int(self.spnEnd.value())
                shotHash["Max"]         = frameMax
                shotHash["Min"]         = frameMin

                self.tableView.setShotData(shotHash, self.tableView.getSelectedRow()[0])
                self.syncSelectionToSourceEditPanel()



    def handleAddSource(self):
        originalRowSelection = self.tableView.getSelectedRow()

        if originalRowSelection:
            indexRow = originalRowSelection[0]

            if indexRow[1]==None:
                indexRow=(indexRow[0]+1, None)
            else:
                indexRow=(indexRow[0], indexRow[1]+1)

        else:
            indexRow = 0

        self._handleAddSourceHelper(originalRowSelection, indexRow)

        self.syncSelectionToSourceEditPanel()
#        self.tableView.setColumnSize()

    def _handleAddSourceHelper(self, originalRowSelection, indexRow, overrideData={}):
        """
        add the source
        """

        self.tableView.insertEmptyRow(indexRow)

        self.tableView.clearSelection()
        self.tableView.selectRow(indexRow)

        self.handleShotSelectionChanged()

        from __init__ import SUPPORTED_IMAGE, SUPPORTED_MOVIE
        image_list = " ".join( ["*.%s" % ext for ext in SUPPORTED_IMAGE] )
        movie_list = " ".join( ["*.%s" % ext for ext in SUPPORTED_MOVIE] )

        filter = "image sequence file (%(image_list)s);; movie (%(movie_list)s);;all files (*.*)" % vars()

        fileNameList = self.promptEditSource(
                                         default=self.txtEditSourcePath.text(),
                                         title="Import Source Sequence - select frame source or movie",
                                         filter=filter,
                                         flgMultiSelection=True
                                         )

        if len(fileNameList)==0: # user cancel import
            self.removeSource() # remove the new empty item

            self.tableView.selectRows(originalRowSelection)
            self.handleShotSelectionChanged()

        else:
            newIndex = indexRow
            fileNameList = [str(item) for item in fileNameList]

            filePathAlreadyAdded = [] # in case the user pick more than one frame from the same source

            for fileName in fileNameList:
                result = get_frame_sequence_source(fileName)

                if result!=None:
                    filePath, fileName, frameMin, frameMax = result
                    fullTemplatePath = (filePath, fileName)

                    if fullTemplatePath in filePathAlreadyAdded:
                        continue
                    else:
                        filePathAlreadyAdded.append(fullTemplatePath)

                    if newIndex!=indexRow:
                        self.tableView.insertEmptyRow(newIndex)

                    # get padding info
                    pad = int ( self.lblDefaultPad.text() )
                    if (int(frameMax) - int(frameMin)) < pad*2:
                        pad=0

                    shotHash = {}
                    shotHash["Source Path"] = filePath
                    shotHash["Source Name"] = fileName
                    shotHash["Start"]       = int(frameMin) + pad
                    shotHash["End"]         = int(frameMax) - pad
                    shotHash["Duration"]    = shotHash["End"] - shotHash["Start"]
                    shotHash["Max"]         = frameMax
                    shotHash["Min"]         = frameMin
                    shotHash["Head"]        = 0
                    shotHash["Tail"]        = 0
                    shotHash["L - Source"]  = "*"

                    # set any override data, this is use when the parent want to pass on some data to children
                    for k in overrideData.keys():
                        shotHash[k] = overrideData[k]

                    self.tableView.setShotData(shotHash, newIndex)

                    # increment the index
                    if type(newIndex)==int:
                        newIndex +=1
                    elif newIndex[1]==None:
                        newIndex = (newIndex[0]+1, None)
                    else:
                        newIndex = (newIndex[0], newIndex[1]+1)

                    #self.tableView.setShotData(shotHash, self.tableView.getSelectedRow()[0])


    def playlistEditTrigger(self):
        self.pnlEditSource.show()
        self.txtEditSourcePath.setFocus()
        self.cmdEditSource.setChecked(True)

    def editSource(self):
        if self.pnlEditSource.isVisible():
            self.pnlEditSource.hide()

        else:
            self.pnlEditSource.show()
            self.txtEditSourcePath.setFocus()

        self.syncSelectionToSourceEditPanel()


    def promptEditSource(self,
                         default="",
                         title="Import Source Sequence - select a frame from sequence",
                         filter=None,
                         flgMultiSelection=False,
                         ):
        '''
        Prompt user for source, if user selects a directory then return True, else returns False.
        '''
        from __init__ import SUPPORTED_IMAGE, SUPPORTED_MOVIE
        image_list = " ".join( ["*.%s" % ext for ext in SUPPORTED_IMAGE] )
        movie_list = " ".join( ["*.%s" % ext for ext in SUPPORTED_MOVIE] )

        if filter==None:
            filter = "image sequence file (%(image_list)s);; movie (%(movie_list)s);;all files (*.*)" % vars()

        sourcePath, sourceName =  os.path.split(str(default))
        if flgMultiSelection:
            fileNameList = QtGui.QFileDialog.getOpenFileNames(self,
                                         self.tr(title),
                                         sourcePath,
                                         self.tr(filter)
                                         )

            return fileNameList
        else:
            fileName = QtGui.QFileDialog.getOpenFileName(self,
                                         self.tr(title),
                                         sourcePath,
                                         self.tr(filter)
                                         )
            if not( fileName.isEmpty() ):
                return fileName


    def closeEditSource(self):
        self.pnlEditSource.hide()
        self.cmdEditSource.setChecked(False)

    def _moveSource(self, direction):
        rowIndexList = [row[0] for row in self.tableView.getSelectedRow()]
        newRowIndexList = map(lambda x: x+direction, rowIndexList)

        # the top row is already at top can't move anymore
        if direction==-1 and newRowIndexList[0] < 0: return
        if direction==1 and newRowIndexList[-1] >= self.tableView.model().rowCount(): return

        # move the data
        self.tableView.selectionModel().clearSelection()
        if direction==1:
            rowIndexList.reverse()

        for rowIndex in rowIndexList:
            # get shot data and it's child data and the expand state
            data = self.tableView.getShotData(rowIndex)
            varData = self.tableView.getVariantShotData(rowIndex)

            flgExpanded = self.tableView.isExpanded(self.tableView.model().index(rowIndex, 0) )

            # now remove the row, it seems to remove the child as well by default
            self.tableView.model().removeRow(rowIndex)

            # now add the shot data
            self.tableView.addShot(data, rowIndex+direction)
            for i in range(len(varData)):
                self.tableView.addShot( varData[i], index=rowIndex+direction, childIndex=i, flgAutoExpand=flgExpanded)


        # reselect the data
        for rowIndex in newRowIndexList:
            self.tableView.selectRow(rowIndex)

    def moveSourceUp(self):
        self._moveSource(-1)

    def moveSourceDown(self):
        self._moveSource(1)

    def removeSource(self):
        ''' Remove the selected items from the table. '''

        # first find the selected item
        smodel = self.tableView.selectionModel()

        rowIndexList = self.tableView.getSelectedRow()
        rowIndexList.reverse()

        for index in rowIndexList:
            self.tableView.removeShotRow(index)

        self.handleShotSelectionChanged()


    def handleEditAudioPrompt(self):
        fileName = self.promptEditSource(
                                         default=self.cboAudioPath.currentText(),
                                         title="Import Audio Source",
                                         filter="Audio (*.wav)"
                                         )

        if fileName!=None:
            self.cboAudioPath.insertItem(0, fileName)
            self.cboAudioPath.setEditText(fileName)



    def clearPlaylist(self):
        reply = QtGui.QMessageBox.warning(self, self.tr("Clear Playlist"),
                                          "Clear the playlist?", self.tr("Ca&ncel"),
                                          self.tr("&Ok"))
        if reply:
            self.tableView.clear()
            self.tableView.setupData()
            self.cboAudioPath.clear()


        self.handleShotSelectionChanged()


    def resetGui(self):

        self.tableView.clear()
        self.tableView.setupData()
        self.cboAudioPath.clear()
        self.tableView.clearThumbCache()


    def _parseSourcePath(self, sourcePath):
        source_path, source_name = os.path.split(sourcePath)
        source_name_token = source_name.split(".")
        if source_name_token[-2].find("#")!=-1:
            source_name_token[-2] = "#"
        source_name = ".".join(source_name_token)
        return source_path, source_name


if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)

    mixMasterShell = MixMasterRichShell(None, None, "/tmp/visione")
    test = "/drd/jobs/hf2/story/sc_100a_rndtestWillgoAway/previs/RV/test.rv"
    test = "/tmp/test.rv"
    mixMasterShell.mixMasterMain.importPlaylist(test)
    mixMasterShell.mixMasterMain.generateRvFile("/tmp/d.rv")

    mixMasterShell.show()
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

