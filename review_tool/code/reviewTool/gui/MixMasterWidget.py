import      sys, os, re, pprint
import      time, calendar

import      tank

import      drGadgets.lib.qtGuiPreference as guiPrefSaver
from        PyQt4 import QtGui, QtCore

from        mixMasterEmbedded.MixMasterGui import MixMasterGui
from        mixMasterEmbedded.SceneTree import FrameThumbnailDelegate
from        reviewTool.projectAwareness import is_tank_render_container
from        reviewTool import projectAwareness

#from        reviewTool.util import *
from        rv_tools.util import get_frame_sequence_source as get_frame_sequence_source_fast
from        reviewTool.util import get_tmp_rv_file_name
from        reviewTool import FullContextSession

import      resource
from        pprint import pprint
from        reviewTool import TANK_VFS_SUCKS

import      sinfo_utils
import      rv_tools

TEST = False

from reviewTool import TANK_VFS_SUCKS
FrameThumbnailDelegate.FRAME_TEXT_WIDTH = 0

FRAME_TEXT_WIDTH = 45  # roughly the width of diplaying 4 digits in size 10 font.

class FrameRangeDelegate(QtGui.QItemDelegate):
    '''
    The class displays the frame image in table cell
    '''
    def __init__(self, parent = None):
        '''
        The class initialize with the parent table view class,
        This is essential because all thumb cache gets stored in parent,
        the delegate class does not store data.
        The path to the image that needs to be drawn comes from the model data.
        '''
        super(FrameRangeDelegate,self).__init__(parent)
    
    def paint (self, painter, option, index):
        super(FrameRangeDelegate,self).drawBackground( painter, option, index );

        minCell     = index.sibling(index.row(), self.parent().headerOrder.index("Min"))
        maxCell     = index.sibling(index.row(), self.parent().headerOrder.index("Max"))
        min         = int(minCell.data().toInt()[0])
        max         = int(maxCell.data().toInt()[0])
        frame       = int ( index.data(QtCore.Qt.DisplayRole).toInt()[0] )

        x1, y1, x2, y2 = option.rect.getCoords()
        h = y2-y1
        w = x2-x1

        if frame > max or frame < min:
            painter.setPen(     QtGui.QPen( QtGui.QColor(0,0,0,0) ) )
            painter.setBrush(   QtGui.QBrush( QtGui.QColor(255,0,0,125) ) )

            painter.drawRect(
                                x1, y1,
                                w, h
                              )

        pal = QtGui.QApplication.palette()
        ############## draw text

        if (option.state & QtGui.QStyle.State_Selected):
            painter.setBrush(  pal.brush(QtGui.QPalette.Active, QtGui.QPalette.HighlightedText) )
            painter.setPen(    QtGui.QPen(pal.color(QtGui.QPalette.Active, QtGui.QPalette.HighlightedText)) )
        else:
            painter.setPen(     QtGui.QPen( pal.color(QtGui.QPalette.Active, QtGui.QPalette.Text)) )
            painter.setBrush(   QtGui.QBrush( pal.color(QtGui.QPalette.Active, QtGui.QPalette.Text)) )


        painter.drawText(
                            x1+10, y1+h/2+5,
                            index.data(QtCore.Qt.DisplayRole).toString()
                          )

FRAME_TEXT_WIDTH = 45  # roughly the width of diplaying 4 digits in size 10 font.
class NoEditDelegate(QtGui.QItemDelegate):
    '''
    The class displays the frame image in table cell
    '''
    def __init__(self, parent = None):
        '''
        The class initialize with the parent table view class,
        This is essential because all thumb cache gets stored in parent,
        the delegate class does not store data.
        The path to the image that needs to be drawn comes from the model data.
        '''
        super(NoEditDelegate,self).__init__(parent)

    def createEditor(self, parent, option, index):
        return None





FRAME_TEXT_WIDTH = 45  # roughly the width of diplaying 4 digits in size 10 font.
class DeptDelegate(QtGui.QItemDelegate):
    '''
    The class displays the frame image in table cell
    '''
    def __init__(self, parent = None):
        '''
        The class initialize with the parent table view class,
        This is essential because all thumb cache gets stored in parent,
        the delegate class does not store data.
        The path to the image that needs to be drawn comes from the model data.
        '''
        super(DeptDelegate,self).__init__(parent)
    
    def setModelData(self, editor, model, index):
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        mmwidget = self.parent()
        while ( mmwidget and not isinstance(mmwidget,MixMasterWidget) ):
            mmwidget = mmwidget.parent()
            
        results = mmwidget.setDepartmentForIndex( index, str(editor.currentText()), self._current_context_shot, ignoreSameDept = True )
        sg_shot,curr_dept,curr_version,new_version,new_dept = results
            
        QtGui.QApplication.restoreOverrideCursor()
        
    def createEditor(self, parent, option, index):
        from reviewTool import projectAwareness
        
        spath = str( index.sibling(index.row(), self.parent().headerOrder.index("Source Path")).data().toString() )
        sname = str( index.sibling(index.row(), self.parent().headerOrder.index("Source Name")).data().toString() )
        tank_rev_id = str( index.sibling(index.row(), self.parent().headerOrder.index("_Tank_Rev_Id")).data().toString() )
        aux_data = str( index.sibling(index.row(), self.parent().headerOrder.index("_Aux_Data")).data().toString() )

        if not aux_data: return

        context_type, scene_id, shot_id, department = aux_data.split(",")
        cshot = FullContextSession.get_project().get_scene(scene_id).get_shot(shot_id)

        self._current_context_shot = cshot
        sib_ver         = index.sibling(index.row(), self.parent().headerOrder.index("Version"))
        sib_dept        = index.sibling(index.row(), self.parent().headerOrder.index("Department"))
        self._ver_now   = str(sib_ver.data().toString())
        self._dept_now  = str(sib_dept.data().toString())

        import copy
        
        widget = self.parent()
        while ( widget and not isinstance(widget,MixMasterWidget) ):
            widget = widget.parent()
        
        dept = copy.copy( widget._context_widget._shown_departments )
        dept.sort()

        cboDept = QtGui.QComboBox(parent)
        cboDept.addItems(dept)

        dept_now = str( index.data().toString() )
        if dept_now in dept:
            cboDept.setCurrentIndex(dept.index(dept_now))

        return cboDept


FRAME_TEXT_WIDTH = 45  # roughly the width of diplaying 4 digits in size 10 font.
class VersionDelegate(QtGui.QItemDelegate):
    '''
    The class displays the frame image in table cell
    '''
    def __init__(self, parent = None):
        '''
        The class initialize with the parent table view class,
        This is essential because all thumb cache gets stored in parent,
        the delegate class does not store data.
        The path to the image that needs to be drawn comes from the model data.
        '''
        super(VersionDelegate,self).__init__(parent)

        self._current_context_shot = None

    def setModelData(self, editor, model, index):
        verCell     = model.itemFromIndex( index )
        parent      = self.parent()
        spathCell   = model.itemFromIndex( index.sibling(index.row(), parent.headerOrder.index("Source Path")) )
        snameCell   = model.itemFromIndex( index.sibling(index.row(), parent.headerOrder.index("Source Name")) )
        sminCell    = model.itemFromIndex( index.sibling(index.row(), parent.headerOrder.index("Min")) )
        smaxCell    = model.itemFromIndex( index.sibling(index.row(), parent.headerOrder.index("Max")) )
        sRangeCell  = model.itemFromIndex( index.sibling(index.row(), parent.headerOrder.index("Source Range")) )
        sRevCell    = model.itemFromIndex( index.sibling(index.row(), parent.headerOrder.index("Revision")) )
        sRevTankIdCell = model.itemFromIndex( index.sibling(index.row(), parent.headerOrder.index("_Tank_Rev_Id")) )

        new_address = str(editor.itemData(editor.currentIndex(), QtCore.Qt.ToolTipRole).toString())
        new_rev_id  = tank.local.Em().find(new_address).get_id()
        new_asset_id = tank.local.Em().find(new_address).get_asset().get_id()

#        print '.......new rev id, ', new_asset_id, ' ', new_rev_id
        new_tank_id_tuple   = "%s,%s" % (new_asset_id, new_rev_id)
        ori_tank_id_tuple   = str( sRevTankIdCell.text() )

        if new_tank_id_tuple==ori_tank_id_tuple: return

#        print '.......ori rev id, ', oriTankId
#
#
#        revAddress  = str( sRevCell.text() )

        dept = str( index.sibling(index.row(), parent.headerOrder.index("Department")).data().toString() )
        ori_ver = str(index.data().toString())
        ver = str(editor.currentText())

#        print '.......ori_address', ori_address
#        print '.............', str(index.data(QtCore.Qt.ToolTipRole).toString())
#        print '......index', dir(index), index.row()
#        print '......new address.......', new_address

#        if (projectAwareness.determine_aux_data_from_tank_address_fast(ori_address)==
#            projectAwareness.determine_aux_data_from_tank_address_fast(new_address)):
#            return

        if TANK_VFS_SUCKS:
            path = tank.find(new_address).system.filesystem_location
        else:
            path = tank.find(new_address).system.vfs_full_paths[0]

        spath, sname, min, max = get_frame_sequence_source_fast(path)
        print '.........path', path
        print '.......sname', sname

        sRevTankIdCell.setText(new_tank_id_tuple)
        spathCell.setText(spath)
        snameCell.setText(sname if sname!=None else "None")
        sminCell.setText(str(min))
        smaxCell.setText(str(max))
        sRevCell.setText(new_address)
        
        mix_master_widget = self.parent()
        while ( mix_master_widget and not isinstance( mix_master_widget, MixMasterWidget ) ):
            mix_master_widget = mix_master_widget.parent()
        
        mix_master_widget.generate_dynamic_data(row_index_list=[index.row()])

        # now emit signal up sync the data to the context view
        mix_master_widget.emit(QtCore.SIGNAL("render_version_updated"), self._current_context_shot, dept, ori_ver, ver)
    
    def checkLatestState( self, index ):
        cshot, current_version, all_versions = self.collectData(index,cachedOnly=True)
        
        # make sure we have some versions
        if ( not all_versions ):
            return -1
        
        # make sure the current version is the first version in the list
        return current_version == all_versions[0][0]
    
    def collectData( self, index, cachedOnly = False ):
        # extract pertinent information from the index
        current_version     = str( index.sibling(index.row(), self.parent().headerOrder.index("Version")).data().toString() )
        aux_data            = str( index.sibling(index.row(), self.parent().headerOrder.index("_Aux_Data")).data().toString() )
        
        # make sure we have auxiliary data
        if ( not aux_data ):
            return (current_version,[],[])
        
        # grab the shot from the auxiliary data
        context_type, scene_id, shot_id, department = aux_data.split(",")
        
        # collect data from the session
        sg_project      = FullContextSession.get_project()
        sg_scene        = None
        sg_shot         = None
        all_versions    = []
        sg_versions     = []
        
        # make sure we have a project
        if ( sg_project ):
            sg_scene    = sg_project.get_scene(scene_id)
        
        # make sure we have a scene
        if ( sg_scene ):
            sg_shot     = sg_scene.get_shot(shot_id)
        
        # make sure we have a shot
        if ( sg_shot ):
            # collect the versions
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            try:
                sg_versions     = sg_shot.get_all_render_from_shotgun(department,cachedOnly=cachedOnly)
                all_versions    = [ (ver["name"],ver["address"].replace("Movie(","Frames(")) for ver in sg_versions ]
            except:
                import traceback
                print 'fail to determine revision versions'
                traceback.print_exc()
            
            QtGui.QApplication.restoreOverrideCursor()
        
        return (sg_shot,current_version,all_versions)
    
    def createEditor(self, parent, option, index):
        # collect the version information
        cshot, current_version, all_versions = self.collectData(index)
        
        # make sure we have some versions to work with
        if ( not all_versions ):
            return
        
        cboVersion = QtGui.QComboBox(parent)
        
        i=0
        for ver, tank_address in all_versions:
            cboVersion.addItem(ver)
            cboVersion.setItemData(i, QtCore.QVariant(tank_address), QtCore.Qt.ToolTipRole)
            i+=1

        self.ver_comment_map = dict(all_versions)

        just_ver_list = [ i[0] for i in all_versions ]
        if current_version in just_ver_list:
            cboVersion.setCurrentIndex(just_ver_list.index(current_version))

        self._current_context_shot = cshot

        return cboVersion
        
    def paint( self, painter, option, index ):
        # check the state of this item
        state = self.checkLatestState(index)
        
        pen = painter.pen()
        
        # if state == -1, then there was no cache data, don't do anything
        # if state == 1, then its in the latest state, draw green background
        # if state == 0, then its not in the lates state, draw red background
        if ( state == 1 ):
            clr = QtGui.QColor('green')
            clr.setAlpha(80)
            
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(clr))
            painter.drawRect(option.rect)
            
        elif ( state == 0 ):
            clr = QtGui.QColor('red')
            clr.setAlpha(80)
            
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(clr))
            painter.drawRect(option.rect)
        
        painter.setPen(pen)
        
        # draw the standard background
        super(VersionDelegate,self).paint(painter,option,index)

#class VersionDelegate(QtGui.QItemDelegate):
#    '''
#    The class displays the frame image in table cell
#    '''
#    def __init__(self, parent = None):
#        '''
#        The class initialize with the parent table view class,
#        This is essential because all thumb cache gets stored in parent,
#        the delegate class does not store data.
#        The path to the image that needs to be drawn comes from the model data.
#        '''
#        QtGui.QItemDelegate.__init__(self, parent)
#        self._parent = parent
#
#        self._current_context_shot = None
#
#    def setModelData(self, editor, model, index):
#        verCell     = model.itemFromIndex( index )
#        spathCell   = model.itemFromIndex( index.sibling(index.row(), self._parent.headerOrder.index("Source Path")) )
#        snameCell   = model.itemFromIndex( index.sibling(index.row(), self._parent.headerOrder.index("Source Name")) )
#        sminCell    = model.itemFromIndex( index.sibling(index.row(), self._parent.headerOrder.index("Min")) )
#        smaxCell    = model.itemFromIndex( index.sibling(index.row(), self._parent.headerOrder.index("Max")) )
#        sRangeCell  = model.itemFromIndex( index.sibling(index.row(), self._parent.headerOrder.index("Source Range")) )
#        sRevCell    = model.itemFromIndex( index.sibling(index.row(), self._parent.headerOrder.index("Revision")) )
#        sRevIdCell  = model.itemFromIndex( index.sibling(index.row(), self._parent.headerOrder.index("_Tank_Rev_Id")) )
#
#        revAddress  = str( sRevCell.text() )
#
#        dept = str( index.sibling(index.row(), self._parent.headerOrder.index("Department")).data().toString() )
#        ori_ver = str(index.data().toString())
#        ver = str(editor.currentText())
#
#        # no change
#        if ori_ver==ver:return
#
#        assetId, revId = str(sRevIdCell.text()).split(",")
#
#        tank.server.Em().get_revision_by_id(int(revId))
#
#        # tank_render_address = revAddress.replace(ori_ver, ver)
#
#        if TANK_VFS_SUCKS:
#            path = tank.find(tank_render_address).system.filesystem_location
#        else:
#            path = tank.find(tank_render_address).system.vfs_full_paths[0]
#
#        spath, sname, min, max = get_frame_sequence_source_fast(path)
#
#        spathCell.setText(spath)
#        snameCell.setText(sname)
#        sminCell.setText(str(min))
#        smaxCell.setText(str(max))
#        sRevCell.setText(tank_render_address)
#        mix_master_widget = self._parent._parent
#        mix_master_widget.generate_dynamic_data(row_index_list=[index.row()])
#
#        # now emit signal up sync the data to the context view
#        mix_master_widget.emit(QtCore.SIGNAL("render_version_updated"), self._current_context_shot, dept, ori_ver, ver)
#
#
#    def createEditor(self, parent, option, index):
#
#        spath   = str( index.sibling(index.row(), self._parent.headerOrder.index("Source Path")).data().toString() )
#        sname   = str( index.sibling(index.row(), self._parent.headerOrder.index("Source Name")).data().toString() )
#        dept    = str( index.sibling(index.row(), self._parent.headerOrder.index("Department")).data().toString() )
#        ver_now = str( index.sibling(index.row(), self._parent.headerOrder.index("Version")).data().toString() )
#        aux_data = str( index.sibling(index.row(), self._parent.headerOrder.index("_Aux_Data")).data().toString() )
#
#        if not aux_data: return
#
#        context_type, scene_id, shot_id, department = aux_data.split(",")
#        cshot = FullContextSession.get_project().get_scene(scene_id).get_shot(shot_id)
#
#        cboVersion = QtGui.QComboBox(parent)
#
#        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
#        try:
#            all_items = [ (item["name"], "" )for item in cshot.get_all_render_from_shotgun(department) ]
#        except:
#            import traceback
#            print "fail to determine revision visions"
#            traceback.print_exc()
#
#
#        QtGui.QApplication.restoreOverrideCursor()
#
#        i=0
#        for ver, comments in all_items:
#            cboVersion.addItem(ver)
#            cboVersion.setItemData(i, QtCore.QVariant(comments), QtCore.Qt.ToolTipRole)
#            i+=1
#
#        self.ver_comment_map = dict(all_items)
#
#        just_ver_list = [ i[0] for i in all_items ]
#        if ver_now in just_ver_list:
#            cboVersion.setCurrentIndex(just_ver_list.index(ver_now))
#
#        self._current_context_shot = cshot
#
#        return cboVersion


'''==========================

      MixMasterWidget

=========================='''

class MixMasterWidget(MixMasterGui):
    def __init__(self, parent, prefSaver, thumbTmpRoot, context_project, context_widget):
        MixMasterGui.__init__(self,
                                parent=parent,
                                prefSaver=prefSaver,
                                thumbTmpRoot=thumbTmpRoot)

        '''
        -----------------------------------------------
        initialize UI
        -----------------------------------------------
        '''
        self._context_widget = context_widget
        self.grpEditFrameHold.hide()
        self.grpPlaylist.setTitle("")
        self.grpHandles.hide()
        self.cboSourceVersion.hide()
        self.label_5.hide()
        self.context_project = context_project
        self.shotgun_playlist_id = None # the playlist id if playlist is loaded from shotgun

        self.cmdRemoveSource.hide()

        self.cmdAddSource.hide()
        self.cmdAddVariantSource.hide()
        self.cmdClearPlaylist.hide()

        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        self.grpPlaylist.setPalette(palette)

        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        self.widget.setPalette(palette)


        self.buttonPanel.hide()
        self.cmdPlaySelected.hide()
        self.cmdPlay.hide()


        self.panelBanner.hide()
        self.grpAudioSource.hide()

        self.layout().setMargin(0)
        self.layout().setSpacing(0)

        # update the header info
        header = self.tableView.header()
        header.setContextMenuPolicy( QtCore.Qt.CustomContextMenu )
        header.customContextMenuRequested.connect( self.showHeaderMenu )
        
        self._rclick_column = 0

        '''----------------------------------
        override MixMaster header order/size
        --------------------------------'''
        self.tableView.setSourceTriggerSignalBlocked(True)
        self.tableView.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)
#        self.tableView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tableView.headerOrder = [
                                        "Source Label",
                                        "Source Name",

                                        "Revision",
                                        "Source Path",
                                        "L - Source",
                                        "R - Source",
                                        "Audio",
                                        "Audio Override",
                                        "Audio Offset",
                                        "_Aux_Data",
                                        "_Tank_Rev_Id",
                                        "Cut Order",
                                        "Min",
                                        "Max",
                                        "Head",
                                        "Tail",
                                        "Department",
                                        "Version",
                                        "Start",
                                        "End",
                                        "Audio Start",
                                        "Cut Range",
                                        "Source Range",
                                        "Updated",
                                        "Artist",
                                        "Preview",
                                        ]

        self.tableView.headerSizeHash = {
                                       "Source Path":100 if TEST else 0,  # take up the rest of the space
                                       "Source Name":200 if TEST else 0,
                                       "Source Label":-50,
                                       "L - Source": 0,
                                       "R - Source": 0,
                                       "Audio":0,
                                       "Audio Override":30,
                                       "Audio Offset":0,
                                       "Audio Start": 0,
                                       "Start":100,
                                       "End":100,
                                       "Revision":200 if TEST else 0,
                                       "Max":50 if TEST else 0,
                                       "Min":50 if TEST else 0,
                                       "Head":0,
                                       "Tail":0,
                                       "Cut Order":100 if TEST else 0,
                                       "_Aux_Data":200 if TEST else 0,
                                       "_Tank_Rev_Id":100 if TEST else 0,
                                        "Cut Range":120,
                                        "Source Range":120,
                                        "Version":100,
                                        "Department":100,
                                        "Updated":150,
                                        "Artist":120,
                                        "Preview":80,
                                        }


        self.tableView.headerDataType = {
                                       "Source Path":str,
                                       "Source Name":str,
                                       "Source Range":str,
                                       "L - Source":str,
                                       "R - Source":str,
                                       "Version":str,
                                       "Audio":str,
                                       "Audio Override":str,
                                       "Audio Offset":int,
                                       "Audio Start": int,
                                       "Start":int,
                                       "End":int,
                                       "Preview":int,
                                       "Revision":str,
                                       "Max":int,
                                       "Min":int,
                                       "Head":int,
                                       "Tail":int,
                                       "Cut Order":str,
                                       "Department":str,
                                       "_Aux_Data":str,
                                       "_Tank_Rev_Id":str,
                                        "Updated":str,
                                        "Artist":str,
                                       }
        import reviewTool.gui.resource
        self.tableView.headerIcon = {
                                     "Audio Override":QtGui.QIcon(reviewTool.gui.resource.getImage("sound.png") ),
                                     "Department":QtGui.QIcon(reviewTool.gui.resource.getImage("context_icon_departments.png") ),
                                     "Cut Range":QtGui.QIcon(reviewTool.gui.resource.getImage("cut_grey.png") ),
                                     }

        self.tableView.headerToolTip = {
                         "Audio Override":"Indicate if an override audio has been specified.  You can override the audio from the source edit panel (click on pencil icon)",
                         "Department":"Department of the publish clip source. (click to edit)",
                         "Version":"Version of the publish clip source. (click to edit)",
                         "Start":"The play start frame the clip.  If highlighted it means the frame index is out of source range. (click to edit)",
                         "End":"The play end frame the clip. If highlighted it means the frame index is out of source range. (click to edit)",
                         "Cut Range": "The latest official cut for the shot",
                         "Source Range": "The range of the source",
                         "Updated": "The published date for a published clip, or the modification date for a local clip",
                         "Artist": "The author of the published clip, or the file owner for a local clip",
                         "Source Label": "Source label, if the source is not published, it will be tagged with [External Source]."
                                        }


        self.tableView.headerLabel = {
                                     "Audio Override":"",
                                     "Source Label": "Source",
                                     "Preview": "",
                                     "Department": " Dept",
                                     "Cut Range": " Cut Range",
                                     }

        self.tableView.delegateHash = {
                                "Department":   DeptDelegate,
                                "Version":      VersionDelegate,
                                "Start":        FrameRangeDelegate,
                                "End":          FrameRangeDelegate,
                                "Source Label": NoEditDelegate,
                                "Audio":        NoEditDelegate,
                                "Audio Override":        NoEditDelegate,
                                "L - Source":   NoEditDelegate,
                                "R - Source":   NoEditDelegate,
                                "Min":          NoEditDelegate,
                                "Max":          NoEditDelegate,
                                "Cut Range":    NoEditDelegate,
                                "Source Range": NoEditDelegate,
                                "_Aux_Data":    NoEditDelegate,
                                "_Tank_Rev_Id": NoEditDelegate,
                                "Updated":      NoEditDelegate,
                                "Artist":       NoEditDelegate,
                                "Preview":      FrameThumbnailDelegate
                             }
        
        self.tableView.setEditableColumns( [
            self.tableView.headerOrder.index('Department'),
            self.tableView.headerOrder.index('Version'),
            self.tableView.headerOrder.index('Start'),
            self.tableView.headerOrder.index('End'),
        ])

        self.tableView.setupData()
#        self.tableView.setColumnSize()

        l = self.horizontalLayout

        '''----------------------------
        button appearances / behaviors
        ----------------------------'''

        self.load_saver_button = QtGui.QToolButton()
        self.load_saver_button.setGeometry(QtCore.QRect(160, 110, 48, 32))
        self.load_saver_button.setMinimumSize(QtCore.QSize(38, 32))
        self.load_saver_button.setMaximumSize(QtCore.QSize(38, 32))
        self.load_saver_button.setIconSize(QtCore.QSize(30, 30))
        self.load_saver_button.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.load_saver_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.load_saver_button.setAutoRaise(True)
        self.load_saver_button.setIcon(QtGui.QIcon(resource.getImage("folder.png")))

        self.img_frame_toggle = QtGui.QToolButton()
        self.img_frame_toggle.setGeometry(QtCore.QRect(160, 110, 48, 32))
        self.img_frame_toggle.setMinimumSize(QtCore.QSize(70, 32))
        self.img_frame_toggle.setMaximumSize(QtCore.QSize(70, 32))
        self.img_frame_toggle.setIconSize(QtCore.QSize(80, 30))
        self.img_frame_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.img_frame_toggle.setAutoRaise(True)
        self.img_frame_toggle.setIcon(QtGui.QIcon(resource.getImage("image_film_switch.png")))
        self.img_frame_toggle.setToolTip("Toggle the selected source between image sequence and movie.")
        self.img_frame_toggle.setEnabled(False)
        self.line_4.setVisible(False)
        self.img_frame_toggle.hide()

        self.select_all_button = QtGui.QToolButton()
        self.select_all_button.setGeometry(QtCore.QRect(160, 110, 48, 32))
        self.select_all_button.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.select_all_button.setAutoRaise(True)
        self.select_all_button.setIcon(QtGui.QIcon(resource.getImage("lightning_add.png")))
        self.select_all_button.setToolTip("Select all")

        l.insertWidget(2, self.img_frame_toggle)
        l.insertWidget(2, self.load_saver_button)
        l.insertWidget(2, self.select_all_button)

        self.cmdPlay.disconnectNotify(QtCore.SIGNAL("clicked()"))


        self.connect ( self.chkAudioSource,
                QtCore.SIGNAL("stateChanged (int)"),
                self.sync_addio_source
                )

        self.connect ( self.select_all_button,
                QtCore.SIGNAL("clicked ()"),
                self._select_all_sources
                )

        self.connect (self.tableView,
                QtCore.SIGNAL("select_all_items"),
                self._select_all_sources
                      )

        self.connect (self.tableView,
                QtCore.SIGNAL("remove_selection"),
                self._remove_selection
                      )

        self.connect ( self.tableView,
                QtCore.SIGNAL("sourceCellClicked"),
                self._handle_cell_clicked
                       )

    def _remove_selection(self):
        self._context_widget.clear_checked()


    def _select_all_sources(self):
        self._context_widget.ignore_context_tree_selection_events = True
        self._context_widget.tree.selectionModel().clear()
        self._context_widget.ignore_context_tree_selection_events = False

        self._context_widget.ignore_mixmaster_selection_events = True

        self.tableView.selectAll()

        self._context_widget.ignore_mixmaster_selection_events = False

        self._context_widget._sync_play_button_state()
        self._context_widget.toggle_play_source("playlist")
        self._context_widget._handle_mix_master_selection_change(load_scene_shot_if_required=False)


    def _handle_cell_clicked(self):
        self._context_widget.toggle_play_source("playlist")

    def determine_latest_audio_for_revision(self, revision_address):

        aux_data = projectAwareness.determine_aux_data_from_tank_address_fast( revision_address )
        context_shot = FullContextSession.get_project().get_scene(aux_data["scene"]).get_shot(aux_data["shot"])

        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        audio               = None
        audio_offset        = 0
        audio_frame_start   = -1

        try:
            self._context_widget.logger.info("Searching for latest publish shot audio file...")
            QtGui.QApplication.processEvents()
            audio_object            = context_shot.get_latest_master_audio(return_tank_object = True)
            audio                   = audio_object.system.vfs_full_paths[0]
            
            # extract information from pipeline data
            pipeline_data        = audio_object.properties.get('pipeline_data',{})
            audio_offset         = pipeline_data.get('start_offset',0)
            audio_frame_start    = pipeline_data.get('start_frame',-1)

        except:
            self._context_widget.logger.info("Warning: can not get latest master audio file")

        QtGui.QApplication.restoreOverrideCursor()
        return (audio,audio_offset,audio_frame_start)

    def _handle_override_audio_for_selection_remove(self):
        for row_index in self.tableView.getSelectedRow():
            dataHash = self.tableView.getShotData(row_index)

            if dataHash["Revision"] and dataHash["Audio Override"]=="*":
                self.tableView.setShotData({"Audio Override":"", "Audio":"", 'Audio Offset': 0, 'Audio Start': -1}, index=row_index)

    def _handle_override_audio_for_selection(self):
        for row_index in self.tableView.getSelectedRow():
            dataHash = self.tableView.getShotData(row_index)

            if dataHash["Revision"] and dataHash["Audio Override"]=="":
                audio, audio_offset, audio_frame_start = self.determine_latest_audio_for_revision(dataHash["Revision"])
                
                print 'loading audio: ', audio, audio_offset, audio_frame_start

                if audio:
                    # where possible use the system path
                    try:
                        audio = os.readlink(audio)
                    except:
                        pass

                    audioHash = {}
                    audioHash["Audio Override"]     = "*"
                    audioHash["Audio"]              = audio
                    audioHash['Audio Offset']       = audio_offset
                    audioHash['Audio Start']        = audio_frame_start

                    self.tableView.setShotData(audioHash, index=row_index)


    def sync_addio_source(self):
        """
        Override the sync and suggest the publish audio file.
        """
        if self.stateSyncing:
            return

        # if it's not check, then suggest the audio to be the publish audio.
        row_selection = self.tableView.getSelectedRow()
        audio               = ''
        audio_offset        = 0
        audio_frame_start   = None

        if self.pnlEditSource.isVisible() and self.chkAudioSource.isChecked() \
                        and row_selection and str(self.txtAudioSource.text())=="":

            dataHash = self.tableView.getShotData(row_selection[0])

            if dataHash["Revision"]:
#                aux_data = projectAwareness.determine_aux_data_from_tank_address_fast( dataHash["Revision"] )
#
#                context_shot = ContextSession.get_project().get_scene(aux_data["scene"]).get_shot(aux_data["shot"])
#
#                QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
#                try:
#                    self._context_widget.logger.info("Searching for latest publish shot audio file...")
#                    QtGui.QApplication.processEvents()
#                    audio = context_shot.get_latest_master_audio()
##                    self._context_widget.logger.clear_status()
#                except:
#                    self._context_widget.logger.info("Warning: can not get latest master audio file")
#                QtGui.QApplication.restoreOverrideCursor()

                audio, audio_offset, audio_frame_start = self.determine_latest_audio_for_revision(dataHash["Revision"])
        
        
        # udpate the audio in the text source and the hash information
        self.txtAudioSource.setText(audio if audio else "")

        audioHash = {}
        audioHash["Audio Override"]     = "*" if self.chkAudioSource.isChecked() else ""
        audioHash["Audio"]              = str(self.txtAudioSource.text())
        audioHash['Audio Offset']       = audio_offset
        audioHash['Audio Start']        = audio_frame_start

        self.tableView.setShotData(audioHash, index=row_selection[0])

        self.overrideAudioPanel.setEnabled(self.chkAudioSource.isChecked())


    def handleShotSelectionChanged(self):
        """
        override the selection handle of mixmasterGui to add enable/disable function
        """
        MixMasterGui.handleShotSelectionChanged(self)
        count_row_selected = len( self.tableView.getSelectedRow())

        self.cmdMoveSourceDown.setEnabled(count_row_selected)
        self.cmdMoveSourceUp.setEnabled(count_row_selected)
        self.img_frame_toggle.setEnabled(count_row_selected)
        self.img_frame_toggle.setEnabled(count_row_selected)
        self.cmdEditSource.setEnabled(count_row_selected)


    def handleEditSourcePrompt(self):
        '''
        When the current source is edited
        '''
        from mixMaster import SUPPORTED_IMAGE, SUPPORTED_MOVIE
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
            result = get_frame_sequence_source_fast(str(fileName))
            if result!=None:
                filePath, fileName, frameMin, frameMax = resultf

                shotHash = {}
                shotHash["Source Path"] = filePath
                shotHash["Source Name"] = fileName
                shotHash["Start"]       = int(self.spnStart.value())
                shotHash["End"]         = int(self.spnEnd.value())
                shotHash["Max"]         = frameMax
                shotHash["Min"]         = frameMin

                self.tableView.setShotData(shotHash, self.tableView.getSelectedRow()[0])
                self.syncSelectionToSourceEditPanel()

        self.generate_dynamic_data()
    
    def top_row_count(self):
        return self.tableView.model().rowCount()

    def get_top_row_data(self, row_index):
        return self.tableView.getShotData(row_index)

    def get_row_all_variant_data(self, row_index):
        return self.tableView.getShotAndVariantData(row_index)

    def save_file(self, filename):
        seq = self.get_sequence_data( compare_mode=False, flg_movie_source=False)
        if len(seq.list_clips())!=0:
            rv_tools.save_file(seq, filename)
            return filename


    def play_all(self, flg_movie_source=True, compare_mode=None, department_order=[], rv_session=None, selection_only=True):
        '''
        will always use the given rv_session to play the source movies,
        though it does not necessarily need to know how many rv_sessions
        are active
        '''
        data = self.get_sequence_data( compare_mode=compare_mode,
                                       flg_movie_source=flg_movie_source,
                                       department_order=department_order,
                                       selection_only=selection_only )

        if rv_session==None:
            rv_session = rv_tools.get_rv_session()

        temp_rv_file = get_tmp_rv_file_name( self.shotgun_playlist_id );

        if compare_mode!=None:
            rv_session.compare(data, mode=compare_mode, temp_rv_file=temp_rv_file)
            rv_session.eval("resizeFit()")
        else:
            rv_session.open(data, temp_rv_file=temp_rv_file)
            rv_session.eval("resizeFit()")


    def get_sequence_data(self, compare_mode=False, flg_movie_source=True, department_order=[], selection_only=False):
        """
        Returns a hash, if compare mode is False, just return "general" sequence
        else, the key will be the department and the value is a sequence.
        """
        audio_offset_list = [1,] # is a makeshift solution for storing the audio start frame
                                 # the audio start should be sync with cut

        seq = rv_tools.edl.Sequence()
        row_count = self.tableView.model().rowCount()

        if selection_only:
            play_index_list = [ r[0] for r in self.tableView.getSelectedRow() ]
        else:
            play_index_list = range(row_count)

        for i in play_index_list:
            self._context_widget.logger.show_progress(len(play_index_list), i, "Generating RV file for selected shots...")

            shot_data = self.tableView.getShotData(i)

            source_tank_obj     = None
            full_path       = None

            if shot_data['Revision']!="": # publish source
                # the default is the frame address
                tank_address = str(shot_data['Revision'])

                if flg_movie_source:
                    tank_address = tank_address.replace('Frames', 'Movie')

                source_tank_obj = tank.find(tank_address)

                if flg_movie_source:
                    if TANK_VFS_SUCKS:
                        full_path = source_tank_obj.system.filesystem_location
                    else:
                        full_path = source_tank_obj.system.vfs_full_paths[0]
                else:
                    full_path = shot_data["Source Path"] + '/' + shot_data["Source Name"]

            else: # local source
                full_path = shot_data["Source Path"] + '/' + shot_data["Source Name"]

            # correct the frame in and frame out if it's a
            frame_in      = shot_data["Start"]
            frame_out     = shot_data["End"]
            frame_min     = shot_data["Min"]
            frame_max     = shot_data["Max"]

            audio_offset = int(shot_data["Audio Offset"]) if shot_data["Audio Offset"] else 0
            audio_frame_start = int(shot_data['Audio Start']) if shot_data['Audio Start'] else -1
            
            # grab the sinfo data from tank
            stereo_pair     = None
            sinfo_path      = ''
            sinfo_data      = {}
            sinfo_address   = shot_data['Revision'].replace('Frames','SInfoFile').replace('Movie','SInfoFile')
            if ( sinfo_address ):
                try:
                    sinfo_path = tank.find(sinfo_address).system.filesystem_location
                except:
                    sinfo_path = ''
                
                if ( sinfo_path and os.path.exists(sinfo_path) ):
                    sinfo_data = sinfo_utils.unpickle_dict(sinfo_path)
            
            if ( sinfo_data ):
                stereo_pair = sinfo_data.get('stereo_pair')
            
            c = seq.add_clip(
                            left        = full_path,
                            frame_in    = frame_in,
                            frame_out   = frame_out,
                            frame_min   = frame_min,
                            frame_max   = frame_max,
                            audio       = shot_data["Audio"] if shot_data["Audio"] else None,
                            use_clip_audio  = True if shot_data["Audio Override"] else False,
                            audio_offset    = audio_offset,
                            audio_frame_start = audio_frame_start,
                            stereo_pair         = stereo_pair
                         )

            if shot_data['Revision']!="": # publish source
                aux = projectAwareness.determine_aux_data_from_tank_address(source_tank_obj, include_shotgun_data=True )
                c.set_meta_data ( "scene_shot", aux["shot"],                        label="Shot",       hidden=False )
                c.set_meta_data ( "department", aux["department"],                  label="Department",     hidden=False )
                c.set_meta_data ( "scene_shot_index", aux["cut_order"],                                     hidden=True )

                c.set_meta_data ( "tank_rev_obj", tank_address,                     label="Tank Revision",  hidden=False )

        self._context_widget.logger.hide_progress()
        if compare_mode:
            seq_list = seq.group_by_department(
                                               department_meta_id = "department",
                                               scene_shot_meta_id = "scene_shot",
                                               row_order_meta_id  = "scene_shot_index",
                                               department_order   = department_order
                                               )
            return seq_list
        else:
            return seq


    def patch_fcp_data(self, clip_list):
        from reviewTool import ContextSession

        for i in range(len(clip_list)):
            clip = clip_list[i]
            source = os.path.split(clip.left_source_path)[-1]

            # expecting something like:
            # 01b_150_lens_v036.mov
            result = re.search("([\w])+_([\w])+_([\w])+_v([\w])+\.mov", source)
            if result:
                scene, shot, dept, version = result.groups()

                tc = projectAwareness.get_tank_render_container(ContextSession.get_project().name,
                                                           scene_name=scene,
                                                           shot_name=shot,
                                                           department=dept
                                                           )

                path = tank.find("Frames(%s, %s))" % (version, tc)).system.filesystem_location
                folder, name, smin, smax = rv_tools.util.get_frame_sequence_source(path)

                clip.left_source_path   = os.path.join(folder, name)
                clip.left_source_in     = smin
                clip.left_source_out    = smax

                clip.frame_in  =  smin - clip.frame_in
                clip.frame_out =  smax - clip.frame_in


    def import_playlist(self, filename):
        rvf = rv_tools.open_file(filename, "r")

        clip_list = rvf.get_sequence().list_clips()

        if filename.endswith(".xml"):
            rv_tools.util.patch_fcp_data(clip_list)

        for i in range(len(clip_list)):
            clip = clip_list[i]

            self._context_widget.logger.show_progress(len(clip_list), i, "Generating RV file for selected shots...")
            sourcePath, sourceName = os.path.split(clip.left_source_path)

            self.tableView.addShot(
                       { "Source Path": sourcePath,
                         "Source Name": sourceName,
                         "Start":       clip.frame_in,
                         "End":         clip.frame_out,
                         "Audio":       clip.audio if clip.audio else "",
                         "Audio Override":  "*" if clip.use_clip_audio else "",
                         "Audio Offset":    clip.audio_offset if clip.audio_offset!=None else 0,
                         'Audio Start':     clip.audio_frame_start if clip.audio_frame_start != None else -1,
                        }
                       )

        self.generate_dynamic_data()
#        self.tableView.setColumnSize()
        self._select_all_sources()

    def select_source_from_aux_data(self, select_aux_data_list, flg_clear=True):
        # format the list to be in a certain format

        st = time.time()
        select_aux_tuple_list = []

        for aux in select_aux_data_list:
            if type(aux)==tank.local.revision.Revision:
                aux_data = projectAwareness.determine_aux_data_from_tank_address(str(aux))

                select_aux_tuple_list.append( ( aux_data["scene"],
                                                aux_data['shot'],
                                                aux_data['department'],
                                                aux_data['rev_name']  ) )
            else:
                for rev_name in aux['rev_name'].split(","):
                    rev_name = rev_name.strip()
                    select_aux_tuple_list.append(
                                                   ( aux['scene'], aux['shot'], aux['department'], rev_name )
                                                    )

        if flg_clear: self.tableView.clearSelection()

        for row_index in range(self.top_row_count()):
            row_data_hash = self.tableView.getShotAndVariantData(row_index)

            for row_index in row_data_hash.keys():
                row_data = row_data_hash[row_index]

                if row_data["Revision"]:
                    row_aux_data = projectAwareness.determine_aux_data_from_tank_address_fast(row_data["Revision"])

                    formated_aux = (row_aux_data['scene'], row_aux_data['shot'], row_aux_data['department'], row_aux_data['rev_name'])

                    if formated_aux in select_aux_tuple_list:
                        self.tableView.selectRow(row_index)

                        parentItem, rowIndex = self.tableView._resolveParentAndRowIndex(row_index)
                        item = self.tableView.model().indexFromItem( parentItem.child(rowIndex) )
                        self.tableView.scrollTo(item)


    def _search_source_insert_row(self, source_cut_order, source_department):
        '''
        Determine the row index to insert to insert the source with the cut order.
        The cut order is given as (scene order, shot order) tuple
        '''
        cut_order_row_index = 0 # where the source should be according the cut order
        source_scene_order, source_shot_order = source_cut_order

        for index in range(self.top_row_count()):
            row_data = self.get_top_row_data(index)

            row_scene_order =  row_shot_order = -1
            if row_data['Cut Order']!="":
                row_scene_order, row_shot_order = [ int(i) for i in row_data['Cut Order'].split(",") ]


            row_department = row_data['Department'] if row_data['Department'] else ""

            if source_scene_order > row_scene_order:
                cut_order_row_index = index + 1

            elif source_scene_order == row_scene_order:
                if source_shot_order > row_shot_order:
                    cut_order_row_index = index + 1

                elif source_shot_order == row_shot_order and row_department and source_department:
                    dept_order = self._context_widget._department_order

                    if dept_order.index(source_department) > dept_order.index(row_department):
                        cut_order_row_index = index + 1
                    else:
                        break
                else:
                    break
            else:
                break

        return cut_order_row_index


    def refresh_shot_revisions(self, update_scene_id, update_shot_id, update_dept_id, rev_sync_list):
        """
        Insert the revisions from tank_revision_address_list into the table.

        * to do:
            so far the Start / End are read for file source, later it'd be good to read from shotgun_cut_info
            the frame range acquired from file source should be used as min max

        if append_mode, then just add to the list,
        if other wise it's insert mode, where it will replace existing shot items with new ones.
        """

        # get all the data for the update_list
        rev_sync_list = [r.replace("Movie(", "Frames(") for r in rev_sync_list]
        rev_sync_aux_list = [ projectAwareness.determine_aux_data_from_tank_address( r, include_shotgun_data=True )
                                                                     for r in rev_sync_list ]
        update_ver_list = [m["rev_name"] for m in rev_sync_aux_list]
        map_update_ver_to_aux = {}
        for d in rev_sync_aux_list:
            map_update_ver_to_aux[d["rev_name"]] = d

        # first get all the related shots in the table, with (row_index, aux_data)
        current_ver_list = []
        map_current_ver_to_row_index = {}
        for i in range(self.top_row_count()):
            row_data = self.tableView.getShotData(i)

            if not row_data['Revision']: continue

            aux_data = projectAwareness.determine_aux_data_from_tank_address_fast( row_data['Revision'] )

            if (aux_data["department"] == update_dept_id
                    and aux_data["scene"]  == update_scene_id
                    and aux_data["shot"]   == update_shot_id ):
                current_ver_list.append(aux_data["rev_name"])
                map_current_ver_to_row_index[aux_data["rev_name"]] = i

        # remove list
        remove_set  = set(current_ver_list).difference(update_ver_list)
        add_set     = set(update_ver_list).difference(current_ver_list)

        remove_row_index = [ map_current_ver_to_row_index[r] for r in remove_set ]
        remove_row_index.sort(reverse=True)

        for i in remove_row_index:
            self.tableView.removeShotRow(i)

        # now add to the list
        context_shot = FullContextSession.get_project().get_scene(update_scene_id).get_shot(update_shot_id)
        cut_info = context_shot.get_cut_info()


        for rev_name in add_set:
            aux_data = map_update_ver_to_aux[rev_name]

            if TANK_VFS_SUCKS:
                system_path = aux_data["tank_asset_rev_obj"].system.filesystem_location
            else:
                system_path = aux_data["tank_asset_rev_obj"].system.vfs_full_paths[0]

            not_used, fileName, min, max = get_frame_sequence_source_fast ( system_path )

            if fileName==None: # ie the frames were not published properly
                min = aux_data["tank_asset_rev_obj"].properties["first_frame"]
                max = aux_data["tank_asset_rev_obj"].properties["last_frame"]


            shot_data_hash = {
                            'Revision'      :   str(aux_data["tank_asset_rev_obj"]),
                            'Source Name'   :   fileName,
                            'Source Path'   :   system_path,
                            'Start'         :   cut_info['sg_cut_start'] if cut_info else min,
                            'End'           :   cut_info['sg_cut_end'] if cut_info else max,
                            'Max'           :   max,
                            'Min'           :   min,
                            }

            insert_row_index = self._search_source_insert_row(aux_data["cut_order_index"], aux_data["department"])
            self.tableView.addShot(shot_data_hash, insert_row_index)
#            self.tableView.selectRow(insert_row_index)
            self.generate_dynamic_data([insert_row_index], aux_data)

    def generate_dynamic_data(self, row_index_list=[], aux_data=None):
        '''
        generate data like source label, cut order, cut range, etc
        the dynamic data is generate purely from look at the source path and source name
        '''
        context_project = self.context_project

        if not row_index_list:
            row_index_list = range(self.top_row_count())

        for top_index in row_index_list:
            row_data = self.tableView.getShotData(top_index)

            full_source_path = os.path.join(row_data["Source Path"], row_data["Source Name"])

            if len(row_index_list)==1 and aux_data:
                pass # do nothing
            else:
                aux_data = projectAwareness.determine_aux_data_from_vfs_path(full_source_path)

            add_data = self._generate_dynamic_data_helper(aux_data, row_data["Source Name"], row_data["Min"], row_data["Max"], full_source_path)

            add_data["Source Path"] = row_data["Source Path"]
            add_data["Source Name"] = row_data["Source Name"]

            self.tableView.setShotData(add_data, top_index)


    def _generate_dynamic_data_helper(self, aux_data, default_source_label, frame_min=None, frame_max=None, full_source_path=None):
        context_project = self.context_project

        new_data_hash={}

        if aux_data:
            if aux_data["scene"]=="General":
                pass

            elif context_project.get_scene(aux_data["scene"]) and context_project.get_scene(aux_data["scene"]).get_shot(aux_data["shot"], no_exception=True):

                c_shot          = context_project.get_scene(aux_data["scene"]).get_shot(aux_data["shot"], no_exception=True)
                if c_shot:
                    cut_range       = c_shot.get_cut_info()
                    cut_range       = "%s-%s" % (cut_range["sg_cut_start"],  cut_range["sg_cut_end"]) if cut_range else ""

                    cut_order       = c_shot.get_cut_order()
                    cut_order       = "%s,%s" % (cut_order[0], cut_order[1])

                    new_data_hash["Cut Order"]      = cut_order
                    new_data_hash["Cut Range"]      = cut_range

            source_label    = aux_data["shot"]

            new_data_hash["Source Label"]   = source_label
            new_data_hash["Department"]     = aux_data["department"]
            new_data_hash["Head"]           = 0
            new_data_hash["Tail"]           = 0

            new_data_hash["Version"]        = aux_data["rev_name"]
            new_data_hash["Source Range"]   = "%s-%s" % (frame_min, frame_max) if frame_min!=-1 else "None"
            new_data_hash["_Aux_Data"]      = "%s,%s,%s,%s" % (aux_data["context_type"],aux_data["scene"],aux_data["shot"],aux_data["department"])
            new_data_hash["_Tank_Rev_Id"]   = "%s,%s" % aux_data["tank_rev_id"]

            new_data_hash["Preview"]        = (frame_min +  frame_max)/2 if frame_min!=None else 0

            tank_obj = tank.local.Em().get_revision_by_id(*aux_data["tank_rev_id"])
            # extract time data
            timetuple = tank_obj.get_creation_date().timetuple()
            # convert to local time
            timetuple = time.localtime(calendar.timegm(timetuple))
            timetuple = QtCore.QDateTime(QtCore.QDate(*timetuple[0:3]), QtCore.QTime(*timetuple[3:6]))

            author = tank_obj.get_property("created_by").get_value()
            authorName=""
            if author!=None:
                authorName = author.get_name()

            new_data_hash["Revision"]     = str( tank_obj )
            new_data_hash["Artist"]       = str( authorName )
            new_data_hash["Updated"]      = str( timetuple.toString("dd/MM/yy h:m:sap") )

        else:
            new_data_hash["Source Label"]   = default_source_label + " [External Source]"
            new_data_hash["Source Range"]   = "%s-%s" % (frame_min, frame_max)  if frame_min!=-1 else "None"
            new_data_hash["Preview"]        = (frame_min +  frame_max)/2

            query_file = None

            if os.path.isfile(full_source_path):
                query_file = full_source_path
            else:
                m_full_source_path = full_source_path.replace(".#.", ".%s." % str(frame_min))

                if os.path.isfile( m_full_source_path ):
                    query_file = m_full_source_path

            if query_file:
                t = time.gmtime( os.path.getmtime(query_file) )
                timetuple = time.localtime(calendar.timegm(t))
                timetuple = QtCore.QDateTime(QtCore.QDate(*timetuple[0:3]), QtCore.QTime(*timetuple[3:6]))
                new_data_hash["Updated"]      = str( timetuple.toString("dd/MM/yy h:m:sap") )

                import pwd

                file_stat = os.stat(query_file)
                u_info = pwd.getpwuid(file_stat.st_uid)
                if u_info:
                    new_data_hash["Artist"]      = u_info[0]


        if not new_data_hash.has_key("Cut Range") or new_data_hash["Cut Range"]=="0-0":
            new_data_hash["Cut Range"]  = "" #new_data_hash["Source Range"]
            new_data_hash["Start"], new_data_hash["End"] = [int(fi) for fi in new_data_hash["Source Range"].split("-")]

        return new_data_hash

    def get_shot_data_with_aux_id(self):
        # iterate through all the rows, find the matching row to remove from
        # at the sametime, also find where it should be according to the order
        all_source_data = []
        for index in range(self.top_row_count()):
            row_data_list = self.get_row_all_variant_data(index).values()

            for row_data in row_data_list:
                fullpath = os.path.join(row_data['Source Path'], row_data['Source Name'])

                row_data["aux_id"] = None
                if row_data["Revision"]:
                    row_data["aux_id"] = projectAwareness.determine_aux_data_from_tank_address_fast(row_data["Revision"])

                all_source_data.append(row_data)

        return all_source_data


    def _moveSource(self, *argList, **argHash):
        """
        Override some function to prevent it from doing sync
        """
        self._context_widget.ignore_mixmaster_selection_events = True
        MixMasterGui._moveSource(self, *argList, **argHash)
        self._context_widget.ignore_mixmaster_selection_events = False
    
    #--------------------------------------------------------------------------------
    #                           header right click methods
    #--------------------------------------------------------------------------------
    
    def rclickSortAscending( self ):
        self.tableView.sortByColumn( self._rclick_column, QtCore.Qt.AscendingOrder )
    
    def rclickSortDescending( self ):
        self.tableView.sortByColumn( self._rclick_column, QtCore.Qt.DescendingOrder )
    
    def rclickSwitchDepartment( self, applyToAll = False ):
        import copy
        depts = copy.deepcopy(self._context_widget._shown_departments)
        depts.sort()
        
        sel_dept, accepted = QtGui.QInputDialog.getItem( self, 'Select Department', 'Switch to...', depts )
        if ( sel_dept and accepted ):
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            
            # select all of the items
            if ( applyToAll ):
                self.tableView.setUpdatesEnabled(False)
                self._select_all_sources()
                self.tableView.setUpdatesEnabled(True)
            
            # make sure all the data is loaded
            self._context_widget.ignore_context_tree_selection_events   = True
            self._context_widget.ignore_mixmaster_selection_events      = True
            
            sel_dept = str(sel_dept)
            
            indexes = self.tableView.selectedIndexes()
            rows    = list(set([index.row() for index in indexes]))
            count   = len(rows)
            
            for i, row in enumerate(rows):
                index = self.tableView.model().index(row,0)
                self._context_widget.logger.show_progress(count,i,'Switching departments...')
                
                self.setDepartmentForIndex( index, sel_dept, ignoreSameDept = False )
            
            self._context_widget.ignore_context_tree_selection_events   = False
            self._context_widget.ignore_mixmaster_selection_events      = False
            
            QtGui.QApplication.restoreOverrideCursor()
    
    def rclickSwitchDepartmentAll( self ):
        self.rclickSwitchDepartment(True)
    
    def rclickSwitchDepartmentSel( self ):
        self.rclickSwitchDepartment(False)
    
    def rclickUpdateVersion( self, applyToAll = False ):
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        # select all of the items
        if ( applyToAll ):
            self.tableView.setUpdatesEnabled(False)
            self._select_all_sources()
            self.tableView.setUpdatesEnabled(True)
        
        # make sure all the data is loaded
        self._context_widget.ignore_context_tree_selection_events   = True
        self._context_widget.ignore_mixmaster_selection_events      = True
        
        indexes = self.tableView.selectedIndexes()
        rows    = list(set([index.row() for index in indexes]))
        count   = len(rows)
        horder  = self.tableView.headerOrder
        
        for i, row in enumerate(rows):
            index = self.tableView.model().index(row,0)
            self._context_widget.logger.show_progress(count,i,'Updating versions...')
            
            # update to the current department
            deptCell    = index.sibling(row, horder.index("Department"))
            curr_dept   = str(deptCell.data().toString())
            
            # update the department
            self.setDepartmentForIndex( index, curr_dept, ignoreSameDept = False )
        
        self._context_widget.ignore_context_tree_selection_events   = False
        self._context_widget.ignore_mixmaster_selection_events      = False
        
        QtGui.QApplication.restoreOverrideCursor()
    
    def rclickUpdateVersionAll( self ):
        self.rclickUpdateVersion(True)
    
    def rclickUpdateVersionSel( self ):
        self.rclickUpdateVersion(False)
    
    def latestVersion( self, sg_shot, dept ):
        from drTank.util import shotgun_session
        
        # create the shotgun session
        sg              = shotgun_session()
        
        # create the filters
        filters = []
        filters.append(['entity','is',{'id':sg_shot.shotgun_field('id'),'type':'Shot'}])
        filters.append(['sg_tank_address','contains','Department(%s)' % dept])
        filters.append(['sg_tank_address','contains','ReviewType(creative)'])       # make sure to use the latest creative render vs. technical
        
        # create the lookup
        columns = [ 'sg_tank_address' ]
        
        # create the order
        order   = [ {'field_name':'created_at','direction':'desc'} ]
        
        # retrieve the shotgun version
        sg_version = sg.api().find_one( 'Version', filters, columns, order )
        
        # ensures that the latest version in shotgun is the one used
        # since there can be discrepencies between the information in tank and
        # in shotgun
        if ( sg_version ):
            address = sg_version['sg_tank_address'].replace('Movie(','Frames(')
            tank_version = tank.find(address)
            if (tank_version):
                return (tank_version.system.name,tank_version.system.filesystem_location)
        return (None,'')
        
    def setDepartmentForIndex( self, index, new_dept, sg_shot = None, ignoreSameDept = True ):
        row         = index.row()
        horder      = self.tableView.headerOrder
        
        deptCell    = index.sibling(row, horder.index("Department"))
        verCell     = index.sibling(row, horder.index("Version"))
        spathCell   = index.sibling(row, horder.index("Source Path"))
        snameCell   = index.sibling(row, horder.index("Source Name"))
        sRangeCell  = index.sibling(row, horder.index("Source Range"))
        sminCell    = index.sibling(row, horder.index("Min"))
        smaxCell    = index.sibling(row, horder.index("Max"))
        auxCell     = index.sibling(row, horder.index("_Aux_Data"))
        
        curr_dept   = str(deptCell.data().toString())
        curr_version= str(verCell.data().toString())
        curr_spath  = str(spathCell.data().toString())
        curr_sname  = str(snameCell.data().toString())
        
        new_version = None
        new_spath   = ''
        new_sname   = ''
        
        # ignore the changes for same departments if desired
        if ( ignoreSameDept and curr_dept == new_dept ):
            return (sg_shot,curr_dept,curr_version,new_version,new_dept)
        
        if ( not sg_shot ):
            aux_data = str(auxCell.data().toString())
            
            # make sure we have aux_data
            if not aux_data:
                return (sg_shot,curr_dept,curr_version,new_version,new_dept)
            
            # extract the shotgun shot info from the aux data
            context_type, scene_id, shot_id, department = aux_data.split(",")
            
            # make sure we have a project
            sg_project = FullContextSession.get_project()
            if ( not sg_project ):
                return (sg_shot,curr_dept,curr_version,new_version,new_dept)
                
            # make sure we have a scene
            sg_scene = sg_project.get_scene(scene_id)
            if ( not sg_scene ):
                return (sg_shot,curr_dept,curr_version,new_version,new_dept)
            
            # make sure we have a shot
            sg_shot = sg_scene.get_shot(shot_id)
            if ( not sg_shot ):
                return (sg_shot,curr_dept,curr_version,new_version,new_dept)
        
        new_version, filepath = self.latestVersion( sg_shot, new_dept )
        if ( not new_version and curr_dept != new_dept ):
            new_version, filepath   = self.latestVersion( sg_shot, curr_dept )
            new_dept                = curr_dept
        
        if ( not new_version ):
            return (sg_shot,curr_dept,curr_version,new_version,new_dept)
        
        spath, sname, min, max  = get_frame_sequence_source_fast(filepath)
        
        # update the model with the latest data
        model = index.model()
        model.itemFromIndex(spathCell).setText(str(spath))
        model.itemFromIndex(snameCell).setText(str(sname))
        model.itemFromIndex(sminCell).setText(str(min))
        model.itemFromIndex(smaxCell).setText(str(max))
        
        self.generate_dynamic_data(row_index_list=[row])
        
        if ( new_version ):
            self.emit(QtCore.SIGNAL('render_version_updated'),sg_shot,curr_dept,curr_version,new_version,new_dept)
        
        return (sg_shot,curr_dept,curr_version,new_version,new_dept)
        
    def showHeaderMenu( self, point ):
        index = self.tableView.header().logicalIndexAt(point)
        
        self._rclick_column = index
        
        # create a new menu
        menu    = QtGui.QMenu(self)
        action  = menu.addAction('Sort Ascending')
        action.setIcon( QtGui.QIcon( resource.getImage('sort_ascending.png') ) )
        action.triggered.connect( self.rclickSortAscending )
        
        action  = menu.addAction('Sort Descending')
        action.setIcon( QtGui.QIcon( resource.getImage('sort_descending.png') ) )
        action.triggered.connect( self.rclickSortDescending )
        
        # determine the exact column
        if ( index != -1 ):
            column = str(self.tableView.header().model().headerData( index, QtCore.Qt.Horizontal ).toString()).strip()
        
        menu.addSeparator()
        
        # add department options
        if ( column == 'Dept' ):
            action = menu.addAction('Switch Selected Departments...')
            action.setIcon( QtGui.QIcon(resource.getImage('switch_department.png')) )
            action.triggered.connect( self.rclickSwitchDepartmentSel )
        
        elif ( column == 'Version' ):
            action = menu.addAction('Update Selected to Latest')
            action.setIcon( QtGui.QIcon(resource.getImage('update_version.png')) )
            action.triggered.connect( self.rclickUpdateVersionSel )
        
        # display the menu
        menu.exec_(QtGui.QCursor.pos())
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

