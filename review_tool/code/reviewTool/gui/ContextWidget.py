from            PyQt4                           import QtGui, QtCore, uic
import          os,sys,re,copy, time, traceback, subprocess
from            pprint                          import pprint
import          pickle, urllib

import          tank
from            tank.common.errors              import *

from            drTank.util                     import shotgun_session
from            MixMasterWidget                 import MixMasterWidget
from            rv_tools.util                   import get_frame_sequence_source as get_frame_sequence_source_fast

from            reviewTool.gui.ShotgunApWidget  import ShotgunApWidget, DepartmentFilterWidget
from            reviewTool                      import ContextSession
from            reviewTool                      import FullContextSession
from            reviewTool                      import projectAwareness
from            reviewTool                      import TANK_VFS_SUCKS

import          resource    # icon images
import          drGadgets.lib.qtGuiPreference   as guiPrefSaver
import          rv_tools
import          rv_tools.settings
import          sinfo_utils

import getpass
import time
from pprint import pprint as pp
import bkdDiagn.sg_bkdDiagn # from breakdown_diagnosis package

guiTemplate = uic.loadUiType(os.path.join(os.path.split(__file__)[0], "resource", "context_widget.ui"))[0]
guiCell     = uic.loadUiType(os.path.join(os.path.split(__file__)[0], "resource", "context_cell.ui"))[0]
shotgunAp   = uic.loadUiType(os.path.join(os.path.split(__file__)[0], "resource", "shotgunApWidget.ui"))[0]

# set the default temp path (in case its not already defined)
os.environ.setdefault('DRD_TEMP','/Local/tmp')

try:
    THUMB_TEMP_ROOT = "/farm/drd/jobs/%s/tmp/review_tool_volatile_icon_cache/" % os.environ.get("DRD_JOB")
    if not(os.path.isdir(THUMB_TEMP_ROOT)):
        os.makedirs(THUMB_TEMP_ROOT)
except:
    THUMB_TEMP_ROOT = os.path.normpath(os.path.expandvars("$DRD_TEMP/review_tool/"))
    if not(os.path.isdir(THUMB_TEMP_ROOT)):
        os.makedirs(THUMB_TEMP_ROOT)

ALL_DEPARTMENTS     = [i.get_name() for i in tank.server.Em().find("Department").get_children()]
ALL_DEPARTMENTS.sort()

DEPARMENT_LABELS    = {   "lens"     : "Lensing",
                          "anim"     : "Animation",
                          "light"    : "Lighting",
                          "crowd"    : "Crowd",
                          "moedit"   : "MoEdit",
                          "flo"      : "Final Layout",
                          "comp"     : "Compositing",
                          "edit"     : "Editing",
                          "rnd"      : "Rnd",
                          "fx"       : "FX",
                          "art"      : "Art",
                          "model"    : "Model",
                          "previs"   : "Previs",
                          "rig"      : "Rig",
                          "skydome"  : "Skydome",
                          "surface"  : "Surface",
                          "visdev"   : "Vis Dev",
                          "charfx"   : "Char FX",
                          "mocap"    : "Mo Cap",
                          "bulkedit" : "Bulk Edit",
                          "charfinal" : "Char Finaling",
                          "stereo" : "Stereo",
                       }

DEPT_DEFAULT_ORDER  = [ "lens", "anim", "light", "crowd", "rnd" ]
DEPT_DEFAULT_SHOWN  = ["anim", "lens", "light","moedit"]

HIDDEN_DEPARMENTS   = [] #["art", "flow", "model", "previs", "rig", "skydome", "surface", "visdev", "charfx", "mocap"]
DEPARTMENTS         = [ d for d in ALL_DEPARTMENTS if d not in HIDDEN_DEPARMENTS ]

SHOT_STATUS_HASH = {
                    'edp':[resource.getIcon("edp.png", as_pixmap=True),"Edit Prep"],
                    'fin':[resource.getIcon("final.png", as_pixmap=True), "Final"],
                    'ip':[resource.getIcon("in_progress.png", as_pixmap=True), "In Progress"],
                    'hld':[resource.getIcon("hold.png", as_pixmap=True), "Hold"],
                    }


def keyPressEvent(event):
    """
    Handle the address modification from the address combo box.
    """
    print '...key pressed', event,str(event.text())

class StatusPrefs(object):
    # describes the options, the order in which they will be saved/loaded, and
    # their default values
    Options     = [ 'enabled', 'order' ]
    Defaults    = [ True,       -1 ]

    def __init__( self, status_hash ):
        self._status_hash  = status_hash
        self._data          = {}

    def parse_string( self, strdata ):
        """
        :remarks::      parses the inputed string information into the
                        user preferences

        :param::        strdata     <str> format: "status2|option1|option2|..;status2|option1|option2"
        """
        # clear the current data cache
        self._data.clear()

        # make sure we have data to load
        strdata = strdata.strip()

        if (strdata):
            # split the status information up
            splt = strdata.split(';')
            for status_data in splt:
                # split the status options
                status_options = status_data.strip().split('|')

                # make sure we have valid data
                if ( status_options ):
                    # first entry will always be the status
                    status = status_options[0]
                    status_data = {}
                    self._data[status] = status_data

                    # subsequent entries will be based on the Options list
                    values = status_options[1:]

                    # extend the saved options with any defaults that have been added
                    # since the last save
                    values += StatusPrefs.Defaults[len(values):]

                    # set the data
                    for i in range(len(values)):
                        try:
                            value = eval(values[i])
                        except:
                            value = values[i]

                        status_data[StatusPrefs.Options[i]] = value

        # extend the data to include all other statuses
        for status in self._status_hash:
            data = dict(zip(StatusPrefs.Options,StatusPrefs.Defaults))
            self._data.setdefault(status,data)

    def set_data( self, data ):
        """
        :remarks::      sets the data for the inputed items based on the inputed dictionary

        :param::        data    <dict> { <str> status: <dict> { <str> option: <variant> value, .. }, .. }
        """
        for status in data:
            status_data = {}

            # filter for proper options
            for option, value in data[status].items():
                if ( not option in StatusPrefs.Options ):
                    continue

                status_data[option] = value

            # update the additional information in the data
            self._data.setdefault(status,{}).update(status_data)

    def set_value( self, status, option, value ):
        """
        :remarks::      sets the current value for the given status option to the inputed value

        :param::        status  <str>
        :param::        option  <str>
        :param::        value   <variant>

        :return::       <bool> success      based on whether the inputed option is valid or not
        """
        # make sure this is a valid option
        if ( not option in StatusPrefs.Options ):
            return False

        self._data.setdefault(status,{})[option] = value

    def to_string( self ):
        """
        :remarks::      records the properties on this preference instance to a saveable string
        :return::       <str>  format:"status1|option1|option2|..;status2|option1|option2|..;"
        """
        output = []
        for status, options in self._data.items():
            # convert the data in the options to a string value for saving
            strdata = [str(options.get(option,StatusPrefs.Defaults[index])) for index, option in enumerate(StatusPrefs.Options)]

            # record the status options as a piped data
            output.append( '%s|%s' % (status,'|'.join(strdata)) )

        # return the converted string data
        strdata = ';'.join(output)
        return strdata

    def value( self, status, option ):
        """
        :remarks::      returns the value for the given option of the inputed status
        :param::        status      <str>
        :param::        option      <str>
        :return         <variant> value
        """
        if ( not option in StatusPrefs.Options ):
            return None

        return self._data.get(status,{}).get(option,StatusPrefs.Defaults[StatusPrefs.Options.index(option)])

    def values( self, option, filter = None ):
        """
        :remarks::      returns a dictionary of values for the given option across all statusses
        :param::        option      <str>
        :param::        filter      <variant>   if supplied, will only return statuses whose option matches the inputed filter
        :return::       <dict> { <str> status: <variant> option, .. }
        """
        if ( not option in StatusPrefs.Options ):
            return fail

        default = StatusPrefs.Defaults[StatusPrefs.Options.index(option)]
        output = {}
        for status in self._status_hash:
            value = self._data.get(status,{}).get(option,default)
            if ( filter == None or value == filter ):
                output[status] = value

        return output

#==========================================================================================
#    ContextCell widget
#
##

class ContextCell(guiCell, QtGui.QWidget):
    ICON_ARROW_DOWN =  None

    def __init__(self, parent, context_widget, context_row_item):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
        self._context_widget     = context_widget
        self._context_row_item  = context_row_item

        if ContextCell.ICON_ARROW_DOWN ==None:
            ContextCell.ICON_ARROW_DOWN = QtGui.QIcon(os.path.join(os.path.split(__file__)[0], "resource", "bullet_arrow_down.png"))

        self.revision_button.setIcon(ContextCell.ICON_ARROW_DOWN)

        self.connect(self.revision_button, QtCore.SIGNAL("clicked()"), self._handle_prompt_revision_selection)

        self.connect(self.checkBox, QtCore.SIGNAL("stateChanged(int)"), self._handle_check_state_change)

        self.flg_exec_check_actions = True

        '''
        ---------------------------------------
        custom attributes
        ---------------------------------------
        '''
        # populated during creation, and passed to other widget when querying
        # for quick indexing from the outside of context view
        self._cut_data_hash = {}
        self._tank_address  = ""

        # used for detect whether this data need to be re-populated from shotgun
        self._isCached      = False
        self._shot_handle   = None

        self._parent_row    = 0
        self._row           = 0
        self._column        = 0
        self._source_aux_id = None # this used to store the id of the context cell to easily identify it among all the other cells
        self._default_version = ""
        self._default_version_data = None

        self._source_order  = ""

        # temp -----------------
        self._sg_cut_info = None
        #-----------------------

    def set_default_version(self, version):
        self._default_version = version
        self._tank_address = self._default_version

    def set_default_version_data(self, default_data):
        self._default_version_data =         default_data


    def reset_default_version(self):
        self.name_label.setText( tank.find(self._default_version).system.name )
        self._tank_address = self._default_version

        if self._default_version_data["sg_data"]["sg_status_1"]:
            self.set_status_icon(self._default_version_data["sg_data"]["sg_status_1"])
        else:
            self.set_status_icon(None)

    def _handle_check_state_change(self, isChecked):
        if self.flg_exec_check_actions:
            self.emit(QtCore.SIGNAL("CellCheckStateChange"), self, isChecked)
            if not isChecked: self.reset_default_version()

    def _handle_prompt_revision_selection(self):
        self._context_widget._handle_prompt_revision_selection(
                                                       cell_widget   = self,
                                                       row_item      = self._context_row_item)

    render_status_hash = {
                        'apr':[resource.getIcon("approved.png", as_pixmap=True),"Approved"],
                        'dap':[resource.getIcon("purple_cow.png", as_pixmap=True), "Dir Approved"],
                        'nfr':[resource.getIcon("icon_x.png", as_pixmap=True), "Not For Review"],
                        'pdirev':[resource.getIcon("outline_cow.png", as_pixmap=True), "Pending Dir Review"],
                        'rev':[resource.getIcon("hand.png", as_pixmap=True), "Pending Review"],
                        'vwd':[resource.getIcon("eye.png", as_pixmap=True), "Viewed"],
                        'fix':[resource.getIcon("thumbdown.png", as_pixmap=True), "Fix Required"],
                        'fcomp':[resource.getIcon("check.png", as_pixmap=True), "Fix Complete"],
                        'techap':[resource.getIcon("techap.png", as_pixmap=True), "Tech Approved"],
                        }

    def set_status_icon(self, status):
        if ContextCell.render_status_hash.has_key(status):
            icon, info = ContextCell.render_status_hash[status]
            self.status_label.setPixmap(icon)
            self.status_label.setToolTip(info)
        else:
            self.status_label.setPixmap(QtGui.QPixmap())



#========================================================================================
#    ContextWidget
#
#
class ContextWidget(guiTemplate, QtGui.QWidget):
    ROW_HEIGHT = 20

    def __init__(self, parent=None, pref_saver=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)

        # create custom properties
        self._shotStatusPrefs   = StatusPrefs(SHOT_STATUS_HASH)
        self._statusPrefs       = StatusPrefs(ContextCell.render_status_hash)
        ContextSession.SORT_STATUS_ORDER = self._statusPrefs.values('order')

        '''
        Source high lighting toggle
        '''

        self.highlight_color = QtGui.QColor(140, 145, 150, 255)

        self.original_highlight_color = self.palette().highlight().color()
        self.hightlighted_text_color = self.palette().highlightedText().color()
        self.original_highlighted_text_color = self.palette().highlightedText().color()
        self._play_source = "playlist"

        p = self.tree_scene.palette()
        p.setBrush(QtGui.QPalette.Active,   QtGui.QPalette.Highlight, QtGui.QBrush(self.highlight_color))
        p.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, QtGui.QBrush(self.highlight_color))
        self.tree_scene.setPalette(p)

        '''-----------
        initialize UI
        -----------'''
        self.set_window_title()
        self.setWindowIcon ( QtGui.QIcon(os.path.join(os.path.split(__file__)[0], "resource", "applications-multimedia.png")) )
        self.tree.clear()
        self.tree.header().setMovable(False)

        if pref_saver:
            self.pref_saver = pref_saver
        else:
            self.pref_saver = guiPrefSaver.QtGuiPreference("/tmp/review_tool.%s.conf" % os.environ.get("USER","general"))


        # assign the logger to the rv_tools
        rv_tools.util.logger = self.logger

        '''-------------------------
        Selection-related attributes
        ------------------------'''
        self._project = FullContextSession.get_project()
        # they store all the scene objects in shotgun order, and all scene names in shotgun order (the latter will be used more often)
        self._all_scene_list = []
#        self._project.list_scene_names() = []
        self._checked_cell_widget = [] # tracks all the check cell widgets

        self.ignore_mixmaster_selection_events = False # to prevent recursive selection events
        self.ignore_context_tree_selection_events = False
        self.ignore_context_tree_check_events = False # when user directly update the playlist

        '''-------------------------------
        RV: single/multiple session toggle
        --------------------------------'''
        # this attribute controls whether to create a new session every time user
        # hits play or to use the active rv session only
        self._rv_singleton = False
        self._rv_session = []


        '''------------------------
        Headers & department labels
        ------------------------'''
        self.tree.header().setClickable(True)

        self._department_order  = DEPARTMENTS
        self._department_order.sort(self._sort_department_to_default)
        self._shown_departments = DEPT_DEFAULT_SHOWN

        self._header_names = ["context", "context_type", "context_meta"] + self._department_order
        self._header_labels = {   "context"  : "Shot/Asset Clips" }
        self._header_labels.update(DEPARMENT_LABELS) # add the department labels

        '''----------------------
        create department columns
        ----------------------'''
        self.tree.header().setResizeMode(0, QtGui.QHeaderView.Stretch)
        self.tree.header().setStretchLastSection(False)

        self.comboBox_scene_input.setFocus()

        SHOT_CELL_WIDTH = 155
        for i in range(len(self._header_names)):
            name = self._header_names[i]
            label = self._header_labels[name] if self._header_labels.has_key(name) else name
            self.tree.headerItem().setText(i, label)

            if name not in self._shown_departments + ["context"]:
                self.tree.hideColumn(i)

            self.tree.setColumnWidth(i, SHOT_CELL_WIDTH)

        # populate both trees
        self._reset_all_scenes()
        self._reset_scenes()
        self._show_given_rows()

        # load the profile combobox
        self.profile_combobox.addItems(rv_tools.settings.profileNames())
        self.profile_combobox.setCurrentIndex(self.profile_combobox.findText(rv_tools.settings.value('current_profile')))


        '''----------
        Icon / Menu
        ----------'''

        self.toolButton_dept_filter.setText("")
        self.toolButton_dept_filter.setIcon(QtGui.QIcon(resource.getImage("context_icon_departments.png")))

        self.status_filter_button.setIcon(QtGui.QIcon(resource.getImage('status_filter.png')))
        self.shot_status_button.setIcon(QtGui.QIcon(resource.getImage('shot.png')))

        self.button_check_select.setIcon(QtGui.QIcon(resource.getImage("context_icon_add.png")))
        self.clear_selected_scene_button.setIcon(QtGui.QIcon(resource.getImage("erase.png")))
        self.refresh_button.setIcon(QtGui.QIcon(resource.getImage("arrow_refresh_big.png")))

        self.button_clear_check.setIcon(QtGui.QIcon(resource.getImage("context_icon_remove.png")))

        self.play_button.setIcon(QtGui.QIcon(resource.getImage("context_icon_play_without_handle.png")))

        self.play_option_button.setIcon(QtGui.QIcon(resource.getImage("cog.png")))

        self.play_option_menu = QtGui.QMenu(self.play_option_button)
        self.compare_option_menu = QtGui.QMenu(self.compare_button)
        self.compare_button.setMenu(self.compare_option_menu)
        self.play_option_button.setMenu(self.play_option_menu)

        self._create_play_option_menu()
        self._create_sort_option_menu()

        icon = QtGui.QIcon()

        icon.addPixmap(QtGui.QPixmap(resource.getImage("image_film_toggle_image.png")),
                                     QtGui.QIcon.Selected,
                                     QtGui.QIcon.On)

        icon.addPixmap(QtGui.QPixmap(resource.getImage("image_film_toggle_film.png")),
                                     QtGui.QIcon.Selected,
                                     QtGui.QIcon.Off)


        self.help_button.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(QtGui.QImage.fromData(resource.HELP_IMAGE, 'png'))))

        self.shotgun_icon = resource.getIcon("shotgun_c.png")


        '''--------------------------------------------
        initialze playlist tab
        --------------------------------------------'''

        self._next_tab_id = 0
        self.playlist_tab_panel.clear()
        self.create_new_tab()
        new_index = self.playlist_tab_panel.addTab(QtGui.QLabel(), "New")

        self.playlist_tab_panel.tabBar().setTabsClosable(True)

        self._ignore_tab_change = False # if the tab is in the process of closing, hence, don't create new tab

        self.playlist_tab_panel.setIconSize(QtCore.QSize(20,20))

        '''-----------------
        custom attributes
        ------------------'''
        self.shotgun_playlist_id    = None # for passing value between loader and mix master, receiving the id that user has picked up

        '''---------
        connections
        ----------'''
        self.connect(
                     self.play_option_menu,
                     QtCore.SIGNAL("aboutToShow ()"),
                     self._handle_show_play_option
                     )

        self.connect(
                     self.play_option_menu,
                     QtCore.SIGNAL("triggered (QAction *)"),
                     self._handle_save_play_option
                     )

        self.connect(
                     self.compare_option_menu,
                     QtCore.SIGNAL("triggered (QAction *)"),
                     self._handle_save_play_option
                     )

        self.connect(
                     self.tree,
                     QtCore.SIGNAL("itemExpanded ( QTreeWidgetItem *  )"),
                     self._handle_expand_scene
                     )

        self.connect(
                     self.tree,
                     QtCore.SIGNAL("clicked (const QModelIndex&)"),
                     self._handle_tree_cell_clicked
                     )

        self.connect(
                     self.tree,
                     QtCore.SIGNAL("itemSelectionChanged ()"),
                     self._handle_tree_cell_selection_change
                     )

        self.connect(
                     self.playlist_tab_panel.tabBar(),
                     QtCore.SIGNAL("tabCloseRequested (int)"),
                     self.close_tab
                     )
        self.connect(
                     self.playlist_tab_panel.tabBar(),
                     QtCore.SIGNAL("currentChanged (int)"),
                     self._handle_tab_change
                     )
        self.connect(
                     self,
                     QtCore.SIGNAL("destroyed ()"),
                     self.handle_destroy)

        self.connect( self.clear_selected_scene_button,
                     QtCore.SIGNAL("clicked()"),
                     self._handle_clear_selected_scene
                     )

        self.connect(
                     self.refresh_button,
                     QtCore.SIGNAL("clicked()"),
                     self._handle_refresh_tool
                     )

        self.connect(
                     self.button_clear_check,
                     QtCore.SIGNAL("clicked()"),
                     self.clear_checked
                     )
        self.connect(
                     self.button_check_select,
                     QtCore.SIGNAL("clicked()"),
                     self.check_selected_cell
                     )
        self.connect(
                     self.tree_scene,
                     QtCore.SIGNAL("itemSelectionChanged()"),
                     self._scene_tree_display_selected
                     )

        self.connect(self.shot_status_button,
                     QtCore.SIGNAL('clicked()'),
                     self.show_shot_status_filter
                     )

        self.connect(self.play_button,
                     QtCore.SIGNAL("clicked()"),
                     self.play_with_method)

        self.connect(self.compare_button,
                     QtCore.SIGNAL("clicked()"),
                     self.compare_with_method)

        self.connect ( self.help_button,
                       QtCore.SIGNAL("clicked()"),
                       self.show_help
                       )
        self.connect( self.profile_combobox,
                      QtCore.SIGNAL('currentIndexChanged(int)'),
                      self.set_current_profile )


        '''----------------------
        scene quick locating
        ----------------------'''
        self.comboBox_scene_input.lineEdit().textChanged.connect(   self._handle_key_pressed_scene_combo )
        self.comboBox_scene_input.lineEdit().installEventFilter(    self )
        self.comboBox_scene_input.activated.connect(                self._apply_combobox_input )

        '''---------------------
        departments filter dialog
        ---------------------'''
        self.connect(
                     self.toolButton_dept_filter,
                     QtCore.SIGNAL("clicked()"),
                     self._department_filter_loader
                     )
        self.connect(
                     self.tree.header(),
                     QtCore.SIGNAL("sectionClicked (int)"),
                     self._select_whole_column
                     )


        # bootstrap some preferences
#        self.pref_saver.bootstrapWidget( widget      = self.checkBox_rv_singleton,
#                                         widgetName  = "checkBox_rv_singleton",
#                                         widgetType  = guiPrefSaver.CHECKBOX, )

        self.pref_saver.bootstrapWidget( widget      = self,
                                         widgetName  = "general_options", )

        self.user_browsed_path = ""

        self.toggle_play_source("context_view")


    def set_current_profile( self, index ):
        self.profile_combobox.blockSignals(True)
        self.profile_combobox.setCurrentIndex(index)
        self.profile_combobox.blockSignals(False)

        # record the change to the settings
        rv_tools.settings.setValue('current_profile', str(self.profile_combobox.currentText()) )

    def set_window_title(self, sessionName=None):
        version = os.environ.get("DRD_CONTEXT_TOOL_VERSION", os.environ.get("REVIEW_TOOL_VER", ""))

        self._main_title = "%sReview Tool - %s [%s]" % (
                                                        "%s - " % sessionName if sessionName else "",
                                                        version, os.environ.get("TANK_PROJECT", "Unknown"))
        self.setWindowTitle(self._main_title)

#    def _handle_tab_double_click(self):
#        print '....tab double click'

    def edit_rv_settings( self ):
        from .rvsettingsdialog import RVSettingsDialog
        if ( RVSettingsDialog.edit(self) ):
            self.set_current_profile(self.profile_combobox.findText(rv_tools.settings.value('current_profile')))
            self.checkBox_rv_singleton.setChecked(rv_tools.settings.value('separate_process'))

    def toggle_play_source(self, play_source=None):
        """
        Switch between playing from playlist or context view
        """
        playlist_p  = self.get_active_mix_master_widget().tableView.palette()
        context_p   = self.tree.palette()

        self._play_source = play_source

        if play_source=="playlist":
            playlist_p.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, QtGui.QBrush(self.original_highlight_color))
            context_p.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, QtGui.QBrush(self.highlight_color))
            playlist_p.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, QtGui.QBrush(self.original_highlight_color))
            context_p.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, QtGui.QBrush(self.highlight_color))

        else:
            playlist_p.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, QtGui.QBrush(self.highlight_color))
            context_p.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, QtGui.QBrush(self.original_highlight_color))
            playlist_p.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, QtGui.QBrush(self.highlight_color))
            context_p.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, QtGui.QBrush(self.original_highlight_color))

        w = self.get_active_mix_master_widget()
        w.tableView.setPalette(playlist_p)
        w.setUpdatesEnabled(False)

        self._sync_play_button_state()

        # make sure to update the look of the table
        w.setUpdatesEnabled(True)

        self.tree.setPalette(context_p)

    def _handle_tree_cell_clicked(self, index):
        self._handle_tree_cell_selection_change()
        self.toggle_play_source(play_source = "context_view")


    def show_help(self):
        ret = subprocess.Popen(["konqueror", "http://prodwiki/mediawiki/index.php/RnD:HF2Projects:ReviewTool"])


    def _create_play_option_menu(self):

        self._play_method_types = [
                                        ("mode_smart_play", "Play Selection (recommended)"),
#                                        ("mode_playlist", "Play from playlist"),
#                                        ("mode_context_cut", "Play from context view : Use cut range"),
                                        ("mode_context_source", "Play from context view : Use source range"),
                                    ]

        self._play_frame_sequences_types = [
                                        ("source_movie", "Use Movie"),
                                        ("source_img_seq", "Use Image Sequence")
                                    ]

        self._play_source_locations = [
                                        ("network_source", "Network Paths"),
                                        ("local_source", "Local Paths")
                                    ]

        self._compare_mode_types = [
                                        #("compare_none",    "Play Normal"),
                                        ("layout",          "Compare Layout Column", resource.getIcon("column.png", opacity=180)),
                                        ("layout_packed",   "Compare Layout Packed", resource.getIcon("packed.png", opacity=180)),
                                        ("stack_blend",     "Compare Stack Blend",   resource.getIcon("blend.png", opacity=180)),
                                        ("stack_wipe",      "Compare Stack Wipe",    resource.getIcon("wipe.png", opacity=180)),
                                    ]

        self._play_frame_sequences = None
        self._play_method = None
        self._play_compare_mode = None
        self._play_source_location = None

        ag = QtGui.QActionGroup(self.play_option_menu)
        for id, label in self._play_method_types:
            a = self.play_option_menu.addAction(label)
            a.setData(QtCore.QVariant(id))
            a.setCheckable(True)
            ag.addAction(a)

        self.play_option_menu.addSeparator ()

        ag = QtGui.QActionGroup(self.play_option_menu)
        for id, label in self._play_frame_sequences_types:
            a = self.play_option_menu.addAction(label)
            a.setData(QtCore.QVariant(id))
            a.setCheckable(True)
            ag.addAction(a)

        self.play_option_menu.addSeparator()

        # waiting for more information before implemeting the resulting actions - # EKH 06/01/11
#        ag = QtGui.QActionGroup(self.play_option_menu)
#        for id, label in self._play_source_locations:
#            a = self.play_option_menu.addAction(label)
#            a.setData(QtCore.QVariant(id))
#            a.setCheckable(True)
#            ag.addAction(a)

#        self.play_option_menu.addSeparator ()

        ag = QtGui.QActionGroup(self.compare_option_menu)
        for id, label,icon in self._compare_mode_types:
            a = self.compare_option_menu.addAction(label)
            a.setData(QtCore.QVariant(id))
            a.setCheckable(True)
            a.setIcon(icon)
            ag.addAction(a)

        # create the RV settings actions
        self.play_option_menu.addSeparator()
        self.play_option_menu.addAction('Edit RV Settings').triggered.connect( self.edit_rv_settings )

    def _create_sort_option_menu( self ):
        # create the sort options
        sort_methods = [ ('sort_by_date','Sort by Date'), ('sort_by_status','Sort by Status') ]

        # create the menu
        menu = QtGui.QMenu(self.status_filter_button)
        ag = QtGui.QActionGroup(menu)
        for id, label in sort_methods:
            a = menu.addAction(label)
            a.setData(QtCore.QVariant(id))
            a.setCheckable(True)
            a.setChecked(ContextSession.SORT_METHOD == id)
            ag.addAction(a)

        menu.addSeparator()

        a = menu.addAction( 'Edit Status Preferences...' )
        a.setIcon(QtGui.QIcon(resource.getImage("cog.png")))
        a.triggered.connect( self.show_status_filter )
        ag.triggered.connect( self.refresh_sort_method )

        # add the menu to the button
        self.status_filter_button.setMenu(menu)

    def loadPreference(self, value):
        """
        This function will be called qt gui preference saver.
        Restores the history and the current path text.
        """
        try:
            scene_tree_size=150
            if value:
                saver =  pickle.loads( urllib.unquote(value) )
                if saver.has_key("window"):
                    self.resize( saver["window"][0], saver["window"][1] )

                if saver.has_key("position"):
                    self.move( saver["position"][0], saver["position"][1] )

                if saver.has_key("department_order") and len(saver["department_order"])==len(self._department_order):
                    self._department_order = saver["department_order"]
                    self._shown_departments = saver["shown_departments"]

                    self._refresh_department_column_state()

                if saver.has_key("user_browsed_path"):
                    self.user_browsed_path = saver["user_browsed_path"]

                if saver.has_key("play_method"):
                    self._play_method = saver["play_method"]

                if saver.has_key('play_source_location'):
                    self._play_source_location = saver['play_source_location']
                    rv_tools.LOCALIZE_SESSION = self._play_source_location == 'local_source'

                if saver.has_key("play_compare_mode"):
                    self._play_compare_mode = saver["play_compare_mode"]

                if saver.has_key("play_frame_sequences"):
                    self._play_frame_sequences = saver["play_frame_sequences"]

                if saver.has_key("splitter_sizes"):
                    scene_tree_size = int(saver["splitter_sizes"])

                if saver.has_key( 'sort_method' ):
                    self.set_sort_method( saver['sort_method'] )

                if saver.has_key( 'status_prefs' ):
                    self._statusPrefs.parse_string(saver['status_prefs'])
                    ContextSession.SORT_STATUS_ORDER = self._statusPrefs.values('order')

                if saver.has_key( 'shot_status_prefs' ):
                    self._shotStatusPrefs.parse_string(saver['shot_status_prefs'])

                if saver.has_key( 'scene_input_list' ):
                    self.reset_scene_input( saver['scene_input_list'].split(',') )

            # ensure the play options is is in one the correct modes
            if self._play_frame_sequences not in [ m[0] for m in self._play_frame_sequences_types ]:
                self._play_frame_sequences = self._play_frame_sequences_types[0][0]

            if self._play_method not in [ m[0] for m in self._play_method_types ]:
                self._play_method = self._play_method_types[0][0]

            if self._play_source_location not in [ m[0] for m in self._play_source_locations ]:
                self._play_source_location = self._play_source_locations[0][0]
                rv_tools.LOCALIZE_SESSION = self._play_source_location == 'local_source'

            if self._play_compare_mode not in [ m[0] for m in self._compare_mode_types ]:
                self._play_compare_mode = self._compare_mode_types[0][0]

            self.compare_button.setIcon([ m[2] for m in self._compare_mode_types if m[0] == self._play_compare_mode][0])

            # load the rv_tools settings
            rv_tools.settings.restore()
            self.set_current_profile(self.profile_combobox.findText(rv_tools.settings.value('current_profile')))
            self.checkBox_rv_singleton.setChecked(rv_tools.settings.value('separate_process'))

            self.splitter.setSizes( [scene_tree_size, self.size().width() - scene_tree_size])

        except Exception, e:
            print "Failed to load preference."
            import traceback
            print traceback.format_exc()


    def savePreference(self):
        """
        This function will be called qt gui preference saver.
        Save the history and the current path text.
        """
        saver = {
                    "window":               ( self.size().width(), self.size().height() ),
                    "position":             ( self.pos().x(), self.pos().y() ),
                    "department_order":     self._department_order,
                    "shown_departments":    self._shown_departments,
                    "user_browsed_path":    self.user_browsed_path,
                    "play_method":          self._play_method,
                    "play_source_location": self._play_source_location,
                    "play_frame_sequences": self._play_frame_sequences,
                    "play_compare_mode":    self._play_compare_mode,
                    "splitter_sizes":       self.splitter.sizes()[0],
                    'sort_method':          ContextSession.SORT_METHOD,
                    'status_prefs':         self._statusPrefs.to_string(),
                    'shot_status_prefs':    self._shotStatusPrefs.to_string(),
                    'scene_input_list':     ','.join([str(self.comboBox_scene_input.itemText(i)) for i in range(self.comboBox_scene_input.count())])
                 }

        # record the rv tools settings
        rv_tools.settings.save()

        return urllib.quote( pickle.dumps(saver) )



    def _get_department_order_index(self, dept):
        if dept in DEPT_DEFAULT_ORDER:
            return "%02d" % (DEPT_DEFAULT_ORDER.index(dept))
        else:
            return "99" + dept


    def _sort_department_to_default(self, x, y):
        return cmp(self._get_department_order_index(x), self._get_department_order_index(y))




    '''
    ==============================

     dealing with show/hide shots

    ==============================
    '''
    def _handle_key_pressed_scene_combo(self, text):
        text                = str(text)
        highlight_color     = QtGui.QColor(255, 218, 51, 255) # brilliant orange self.original_highlight_color # brilliant orange
        default_color       = self.tree_scene.palette().brush(QtGui.QPalette.Active,QtGui.QPalette.Base).color() #QtGui.QColor(190, 190, 190, 255)

        length = len(self._project.list_scene_names())

        self.tree_scene.setUpdatesEnabled(False)
        first = None
        for i in range(length):
            item        = self.tree_scene.topLevelItem(i)
            itemText    = str(item.text(0))

            if text != '' and (itemText.startswith(text) or text.startswith(itemText)):
                item.setBackgroundColor(0, highlight_color)
                if ( not first ):
                    first = item
            else:
                item.setBackgroundColor(0, default_color)

        if ( first ):
            self.tree_scene.scrollToItem(first)

        self.tree_scene.setUpdatesEnabled(True)

    def eventFilter( self, object, event ):
        if ( object == self.comboBox_scene_input.lineEdit() and event.type() == QtCore.QEvent.KeyPress ):
            if ( event.key() in (QtCore.Qt.Key_Enter,QtCore.Qt.Key_Return) ):
                self._apply_combobox_input()

        return False

    def _apply_combobox_input(self):
        scene_name = str(self.comboBox_scene_input.currentText())

        # make sure we have a scene to lookup
        if ( not scene_name ):
            self.tree_scene.selectionModel().clearSelection()
            return

        found = False

        # apply the scene name selection
        default_color       =      self.tree_scene.palette().brush(QtGui.QPalette.Active,QtGui.QPalette.Base).color()
        for i in range(self.tree_scene.topLevelItemCount()):
            item = self.tree_scene.topLevelItem(i)

            # select the particular item
            if ( item.text(0) == scene_name ):
                item.setSelected(True)
                item.setBackgroundColor(0,default_color)
                found = True

        if ( found ):
            # add to the options
            queue = [scene_name]
            for i in range(self.comboBox_scene_input.count()):
                itemText = self.comboBox_scene_input.itemText(i)
                if ( itemText != scene_name ):
                    queue.append(itemText)

            self.reset_scene_input(queue[:10])

    def reset_scene_input( self, items ):
        # block the signals
        self.comboBox_scene_input.lineEdit().blockSignals(True)
        self.comboBox_scene_input.blockSignals(True)

        # clear the items and rebuild them
        self.comboBox_scene_input.clear()
        self.comboBox_scene_input.addItems(items)
        self.comboBox_scene_input.clearEditText()

        # restore the signals
        self.comboBox_scene_input.lineEdit().blockSignals(False)
        self.comboBox_scene_input.blockSignals(False)

    #    isolate the given row(s) from others by hiding the entire tree except for the given row(s)
    def _show_given_rows(self, index_list = None, show_all = False):
        default_color = self.tree_scene.palette().brush(QtGui.QPalette.Active,QtGui.QPalette.Base).color()
        length = len(self._project.list_scene_names())

        if show_all == True:
            for i in range(length):
                self.tree_scene.invisibleRootItem().child(i).setBackgroundColor(0,default_color)
                self.tree.setRowHidden( i , self.tree.rootIndex() , False )
            return
        elif index_list == None or len(index_list) == 0:
            for i in range(length):
                self.tree_scene.invisibleRootItem().child(i).setBackgroundColor(0,default_color)
                self.tree.setRowHidden( i , self.tree.rootIndex() , True )

            return
        else:
            for i in range(length):
                self.tree_scene.invisibleRootItem().child(i).setBackgroundColor(0,default_color)
                if i not in index_list:
                    self.tree.setRowHidden( i , self.tree.rootIndex() , True )
                else:
                    self.tree.setRowHidden( i , self.tree.rootIndex() , False )
                    root_item = self.tree.invisibleRootItem().child(i)
                    if not self.tree.isItemExpanded(root_item):
                        self.tree.expandItem(root_item)
            return

    #    evoke _show_given_rows()
    def _scene_tree_display_selected(self):
        selected_items = self.tree_scene.selectedItems()
        selected_indexes = [
                            self._project.list_scene_names().index(  str(i.text(0))  ) for i in selected_items
                            ]
        self._show_given_rows(selected_indexes, show_all = False)


    '''
    ============================================

        dealing with context row / cell

    ============================================
    '''
    #    get all movie revision
    def _list_revision(self, context_type, context, context_meta, department, revision_type="Movie"):
        if context_type=="scene":
            pass

        elif context_type=="shot":
            shot_name = context
            # context_meta the context meta should be comma delimted string "key=value" pairs
            meta_tokens = context_meta.split(",")
            meta_hash = dict( [token.split("=") for token in meta_tokens ] )

            scene = self._project.get_scene(meta_hash["scene"])
            shot = self._project.get_scene(meta_hash["scene"]).get_shot(shot_name)

            all_render_version = scene.get_all_shot_render_fast(shot_name, department)

            return all_render_version


    def _handle_tree_cell_selection_change(self):
        if self.ignore_context_tree_selection_events: return

        all_aux_data=[]
        for eachSelectedIndex in self.tree.selectionModel().selectedIndexes():
            row_index_list = [ eachSelectedIndex.parent().row(), eachSelectedIndex.row()]
            column_index = eachSelectedIndex.column()

            cell = self._get_cell_widget(row_index_list, column_index)
            if not type(cell) == ContextCell:
                continue

            #cell._source_aux_id {'context_type': 'shot', 'shot': '21a_030', 'scene': '21a'}
            aux_data = cell._source_aux_id
            aux_data["department"] = self._header_names[column_index]
            aux_data["rev_name"] = str( cell.name_label.text() )

            all_aux_data.append(aux_data)

        mm = self.get_active_mix_master_widget()

        self.ignore_mixmaster_selection_events = True
        mm.select_source_from_aux_data(all_aux_data)
        mm.handleShotSelectionChanged()

        self.toggle_play_source(play_source="context_view")

        self.ignore_mixmaster_selection_events = False




    def _action_menu_hovered(self, action):
        tip = action.toolTip()
#        self.logger.info(tip)

    def _handle_prompt_revision_selection(self, cell_widget, row_item):
        '''
        Show all the revisions when user click on a department shot cell.
        '''
        menu = QtGui.QMenu(self)
        import time
        t = time.time()

        context_data = self._get_row_data(row_item, ["context_type", "context", "context_meta"])

        column_index = self.cell_column_index(cell_widget, row_item)
        context_data["department"] = self._header_names[column_index]

        # get the checked version list from the cell_widget
        ver_check_list = str( cell_widget.name_label.text() ).split(", ")

        all_rev = self._list_revision(**context_data)

        # filter revision based on enabled statuses
        if ( ContextSession.SORT_METHOD == 'sort_by_status' ):
            enabled_statuses = self._statusPrefs.values('enabled',True)
            all_rev = [ rev for rev in all_rev if rev['sg_data']['sg_status_1'] in enabled_statuses or not rev['sg_data']['sg_status_1'] ]

        QtCore.QObject.connect(menu, QtCore.SIGNAL("hovered(QAction *)"), self._action_menu_hovered)

        from RevMenuDialog import RevMenuDialog
        rmd = RevMenuDialog(self)

        rmd.setWindowFlags(QtCore.Qt.Popup)

        for rev in all_rev:
            menuItem = rmd.add_item()
            menuItem.rev_name_label.setText(rev["name"])
            menuItem.rev_name_label.setToolTip(rev["name"] + ' (%s)' % rev['review_type'])
            menuItem._meta_data = "%s|@|%s|@|%s" % ( rev["name"], rev["address"], rev["sg_data"]["sg_status_1"] )

            menuItem._shotgun_id = rev["sg_data"]["id"]

            if rev["name"] in ver_check_list and cell_widget.checkBox.isChecked():
                menuItem.name_checkbox.setChecked(True)

            comments = rev["sg_data"]["sg_comments"].replace("\n","") if rev["sg_data"]["sg_comments"] else ""
            if len(comments) > 40:
                menuItem.comment_label.setText(comments[:37] + "..")
            else:
                menuItem.comment_label.setText(comments)

            menuItem.comment_label.setToolTip(comments)

            user_name           = rev["sg_data"]["user"]["name"]
            if len(user_name) > 14:
                menuItem.artist_label.setText(user_name[:12] + "..")
            else:
                menuItem.artist_label.setText(user_name )

            menuItem.artist_label.setToolTip( user_name )

#            if rev["address"].find("technical")!=-1:
#                menuItem.set_render_type("technical")


            comments = rev["sg_data"]["sg_comments"].replace("\n","") if rev["sg_data"]["sg_comments"] else ""
            if len(comments) > 36:
                menuItem.comment_label.setText(comments[:34] + "..")
            else:
                menuItem.comment_label.setText(comments)

            menuItem.date_label.setText(rev["sg_data"]["created_at"].strftime("%d/%m/%y"))

            if rev["sg_data"]["sg_status_1"] and ContextCell.render_status_hash.has_key(rev["sg_data"]["sg_status_1"]):
                render_icon, render_info = ContextCell.render_status_hash[ rev["sg_data"]["sg_status_1"]  ]
                menuItem.lblIcon.setPixmap(render_icon)
                menuItem.lblIcon.setToolTip(render_info)

        # now prompt the user for updates via the popup menu
        rmd.set_user_modified(False)
        rmd.setMinimumWidth(600)
        rmd.setMinimumHeight(400)
        x = cell_widget.pos().x() + ( cell_widget.width() - menu.sizeHint().width())
        p = QtCore.QPoint(x-600, cell_widget.pos().y() + cell_widget.size().height()*2)
        rmd.move(self.tree.mapToGlobal(p))

        rmd.exec_()

        # determine if the user has modified the selection at all
        if ( rmd.is_user_modified() ):
            # handle the user menu rev selection, if any
            selected_rev_label_list = []
            selected_rev_address_list = []
            selected_rev_render_status_list = []

            # parse the data
            for checked_item in rmd.list_checked_items():
                rev_text = checked_item._meta_data
                rev_label, rev_address, render_status = rev_text.split("|@|")

                selected_rev_label_list.append(rev_label)
                selected_rev_address_list.append(rev_address)
                selected_rev_render_status_list.append(render_status)

            cell_widget.checkBox.setChecked( len(selected_rev_label_list)>0 )

            if len(selected_rev_label_list)>0:
                # first one determines the icon
                cell_widget.set_status_icon(selected_rev_render_status_list[0])

                cell_widget.name_label.setText(", ".join(selected_rev_label_list))
                cell_widget._tank_address = "|@|".join(selected_rev_address_list)
            else:
                cell_widget.reset_default_version()


            self._sync_context_tree_to_mixmaster(cell_widget, isChecked=True)


    def _sync_context_tree_to_mixmaster(self, cell_widget, isChecked):
        '''
        sync from the context tree to mix master, as result of some something being checked.
        '''

        #-----------------------------------------------------
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        #-----------------------------------------------------


        all_version_list = []
        if cell_widget.checkBox.isChecked() and cell_widget._tank_address.strip() != "":
            all_version_list += cell_widget._tank_address.split("|@|")

        mmw = self.get_active_mix_master_widget()

        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        try:
            self.ignore_mixmaster_selection_events = True
            mmw.refresh_shot_revisions(         update_scene_id = cell_widget._source_aux_id["scene"],
                                                update_shot_id  = cell_widget._source_aux_id["shot"],
                                                update_dept_id  = cell_widget._source_aux_id["department"],
                                                rev_sync_list   = all_version_list
                                                )
            # sync the selection
            if cell_widget in self._get_selected_cell_from_context():
                for ver in all_version_list:
                    aux_data = cell_widget._source_aux_id
                    aux_data["rev_name"] = projectAwareness.determine_aux_data_from_tank_address_fast(ver)["rev_name"]
                    mmw.select_source_from_aux_data([aux_data], flg_clear=False)

            self.ignore_mixmaster_selection_events = False
        except:
            print 'failed to refresh shot', all_version_list
            traceback.print_exc()

        QtGui.QApplication.restoreOverrideCursor()

        # update the list of check cell widgets
        if isChecked and not(cell_widget in self._checked_cell_widget):
            self._checked_cell_widget.append(cell_widget)

        elif not(isChecked) and cell_widget in self._checked_cell_widget:
            self._checked_cell_widget.remove(cell_widget)

        #-----------------------------------------------------
        QtGui.QApplication.restoreOverrideCursor()
        #-----------------------------------------------------

    def _sync_mixmaster_to_context_tree(self):
        '''---------------------------
        clear the _checked_cell_widget
        ---------------------------'''
        self.clear_checked(clearAll=True)

        # parse through the mix master data
        all_source = self.get_active_mix_master_widget().get_shot_data_with_aux_id()

        # first find all the data that has relavant aux id, only ones that publish has aux id
        # aux id look like {"context_type":"shot", "context":"900_123", "department":"lens"} ... etc
        sync_source_hash = {}
        scene_list = []

        #-----------------------------------------------------
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        #-----------------------------------------------------

        for source in all_source:
            if source["aux_id"]:

                key = (source["aux_id"]["scene"], source["aux_id"]["shot"], source["aux_id"]["department"])
                scene_list.append(source["aux_id"]["scene"])
                if not sync_source_hash.has_key(key):
                    sync_source_hash[key] = []

                sync_source_hash[key].append(source)


        # now sync the data, there are two data to set the check, and the label of what's selected
        shot_item = None
        new_checked = []
        for k in sync_source_hash.keys():
            source_list  = sync_source_hash[k]
            scene_id, shot_id, department = k

            shot_item = self.find_tree_item_from_context(scene_id = scene_id, shot_id=shot_id)

            if shot_item and department in self._shown_departments:
                rev_list = [ source["aux_id"]["rev_name"] for source in source_list ]
                rev_address_list = [ source["Revision"] for source in source_list ]

                cell = self.tree.itemWidget( shot_item, self._header_names.index(department) )

#                cell.flg_exec_check_actions = False # this is set false so that it becomes a passive event and does set of any action
                new_checked.append(cell)
                # cell.checkBox.setChecked(True)
                cell.name_label.setText( ", ".join(rev_list) )
                cell._tank_address = "|@|".join(rev_address_list)
                '''------------------------
                refill _checked_cell_widget
                ------------------------'''
                #self._checked_cell_widget.append(cell)

#                cell.flg_exec_check_actions = True
        unchecklist = set(self._checked_cell_widget).difference(set(new_checked))
        checklist = set(new_checked).difference(set(self._checked_cell_widget))

        for cell in unchecklist:
            cell.flg_exec_check_actions = False
            self.clear_checked(cell)
            self._checked_cell_widget.pop(cell)
            cell.flg_exec_check_actions = True

        for cell in checklist:
            cell.flg_exec_check_actions = False
            cell.checkBox.setChecked(True)
            self._checked_cell_widget.append(cell)
            cell.flg_exec_check_actions = True

        # now scroll the tree
        if shot_item:
            self.tree.scrollToItem(shot_item, QtGui.QAbstractItemView.PositionAtTop)

        #-----------------------------------------------------
        QtGui.QApplication.restoreOverrideCursor()
        #-----------------------------------------------------


    def _handle_expand_scene(self, parent, force = False):
        '''
         1> check if the child has even been populated
         2> be aware that "shot" refer to shot name, a string; "nShot" is the a shot object
            the same apply to scene / nScene as well
         3> cache shot
        '''
        #-----------------------------------------------------
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        #-----------------------------------------------------
        import time
        ctime = time.time()

        if (parent.childCount()==1 and parent.child(0).text(0)=="dummy") or force:
            # clear out the children
            while (parent.childCount()):
                parent.removeChild(parent.child(0))

            # now add the
            c = self._get_row_data(parent, ["context", "context_type"])

            if c["context_type"]=="scene":
                scene = c["context"]

                self.logger.info("Loading clips for scene %s  from shotgun..." % scene )

                nScene = self._project.get_scene(scene)

                # filter shots based on status options
                enabled_statuses = self._shotStatusPrefs.values('enabled',True)
                all_shots = []
                for sg_shot in nScene.list_shots():
                    sg_status = None
                    try:
                        sg_shot.shotgun_field('sg_status_list')
                    except:
                        pass

                    # include non-production shots by default, or filtered production shots
                    if ( not sg_status or sg_status in enabled_statuses ):
                        all_shots.append(sg_shot)
                    elif ( sg_status ):
                        print 'filtered out', sg_status

                all_scene_movie_hash = nScene.get_all_scene_render_shotgun()


                # collect the enabled status list
                enabled_statuses = self._statusPrefs.values('enabled',True)

                i = 0
                length = len(all_shots)

                for nShot in all_shots:
                    # get latest render for every department
                    row_data = {
                                    "context"       : nShot.name,
                                    "context_type"  : "shot",
                                    "context_meta"  : "scene=%s" % scene,
                                    }
                    item = QtGui.QTreeWidgetItem(parent)
                    sizeHint = QtCore.QSize()
                    sizeHint.setHeight(ContextWidget.ROW_HEIGHT)
                    item.setSizeHint(0, sizeHint)


                    latest_mov = None
                    for dept in self._shown_departments:

                        if ( all_scene_movie_hash.has_key(dept) and
                             all_scene_movie_hash[dept].has_key(nShot.name) and
                             all_scene_movie_hash[dept][nShot.name]):

                            # sort based on the latest options
                            movie_list = all_scene_movie_hash[dept][nShot.name]
                            movie_list.sort(ContextSession._sort_shotgun_clips)

                            # filter out disabled statuses
                            if (ContextSession.SORT_METHOD == 'sort_by_status'):
                                movs = [ mov for mov in movie_list if mov['sg_data']['sg_status_1'] in enabled_statuses or not mov['sg_data']['sg_status_1'] ]
                            else:
                                movs = movie_list

                            # grab the latest movie (if one exists based on the filters)
                            if ( movs ):
                                latest_mov = movs[0]
                            else:
                                continue

                        else:
                            continue  # no latest, don't need to create cell widget, save time.

                        cell_widget = ContextCell(self.tree, context_widget=self, context_row_item = item)
                        cell_widget._row = all_shots.index(nShot)
                        cell_widget._parent_row = self._project.list_scene_names().index(scene)
                        cell_widget._column = self._header_names.index(dept)
                        cell_widget._source_aux_id = {"context_type":"shot", "scene": scene, "shot": nShot.name,"department":dept}
                        cell_widget._source_order  = list(nShot.get_cut_order())

                        self.tree.setItemWidget(item, cell_widget._column, cell_widget)

                        cell_widget.name_label.setText( latest_mov["name"] )

                        cell_widget.set_status_icon(latest_mov["sg_data"]["sg_status_1"])
                        cell_widget.set_default_version( latest_mov["address"] )
                        cell_widget.set_default_version_data(latest_mov)


                        cell_widget._tank_address = latest_mov["address"]

                        # cache the shot object handle!!
                        cell_widget._shot_handle = nShot

                        self.connect(cell_widget, QtCore.SIGNAL("CellCheckStateChange"), self._handle_check_update)


                    self._set_row_data(item, row_data)
                    self.logger.show_progress(length, i, "Listing shots...")
                    i += 1

                    QtGui.QApplication.processEvents()

        self._sync_mixmaster_to_context_tree()
        #-----------------------------------------------------
        QtGui.QApplication.restoreOverrideCursor()
        #-----------------------------------------------------

    def _handle_check_update(self, *argList, **argHash):
        if not self.ignore_context_tree_check_events:
            self._sync_context_tree_to_mixmaster(*argList, **argHash)

    def _handle_clear_selected_scene(self):
        self.tree_scene.selectionModel().clear()

    def _handle_refresh_tool(self, prompt_first=True):
        refresh=True
        if prompt_first:
            reply = QtGui.QMessageBox.question(self, "Reload Cached Data",
                                           "Do you wish to reload? (any cached data will be cleared)",
                                           QtGui.QMessageBox.No,
                                           QtGui.QMessageBox.Yes,
                                           )

            if reply == QtGui.QMessageBox.No:
                refresh=False

        if refresh:
            # store the selected items
            selection = [ str(item.text(0)) for item in self.tree_scene.selectedItems() ]

            self.tree_scene.selectionModel().clear()
            self.tree.clear()

            self._checked_cell_widget = []
            self._reset_all_scenes()
            self._show_given_rows([])

            FullContextSession.get_project().clear_shotgun_render_cache()

            # restore the selected items
            self.tree_scene.blockSignals(True)
            for i in range( self.tree_scene.topLevelItemCount() ):
                item = self.tree_scene.topLevelItem(i)
                if ( str(item.text(0)) in selection ):
                    item.setSelected(True)
            self.tree_scene.blockSignals(False)

            # now update after selecting all the items
            w = self.playlist_tab_panel.currentWidget()
            w.setUpdatesEnabled(False)
            self._scene_tree_display_selected()
            w.setUpdatesEnabled(True)

    def _reset_all_scenes(self):
        row = 0

        for scene in self._project.list_scene_names():
            # add item
            item = QtGui.QTreeWidgetItem(self.tree)
            item.setFirstColumnSpanned(True)
            item.setFlags( QtCore.Qt.ItemIsEnabled )

            # add dummy child
            item = QtGui.QTreeWidgetItem(item)
            item.setText(0, "dummy")

            self._set_row_data(
                              self.tree.topLevelItem(row),
                              {"context"       : scene,
                               "context_type"  : "scene",   }
                             )
            row +=1


    #    initialized the tree_scene widget
    def _reset_scenes(self):
        row_widget_list = []
        for scene in self._project.list_scene_names():
            # add item
            row_widget_list.append(
                                   QtGui.QTreeWidgetItem([scene, ])
                                   )
        self.tree_scene.addTopLevelItems( row_widget_list )


    '''
    ===================================

      dealing with row/cell indexing

    ===================================
    '''
    #    input cell widget, return a column index
    def cell_column_index(self, cell_widget, row_item):
        # from the column index, figure out the department.
        return [ self.tree.itemWidget(row_item, i) for i in range(len(self._header_names)) ].index(cell_widget)

    #    showing all the shots
    def expand_row(self, context, context_type="scene"):
        item_list = self._find_row({"context_type":context_type, "context":context})

        if item_list:
            if self.tree.isItemExpanded(item_list[0]):
                return
            else:
                #print item_list[0].treeWidget().indexFromItem(item_list[0]).row(), "EXP"
                self.tree.expandItem(item_list[0])

        else:
            return

    #    return row item
    #    if parent_item given, search the children items
    def _find_row(self, search_hash=None, parent_item=None):
        if parent_item == None:
            parent_item = self.tree.invisibleRootItem()

        result = []
        for i in range(parent_item.childCount()):
            item = parent_item.child(i)

            row_hash = self._get_row_data(item, search_hash.keys())
            #print search_hash, '.....', row_hash
            if search_hash == row_hash:
                result.append(item)
        return result


    def find_tree_item_from_context(self, scene_id, shot_id=None):
        '''
        Based on some criterial, find in the treewidget the tree item that matches the criteria.
        Assumption: there criterial uniquely identifies a tree item.  Ie, there can't be two shot with same name in same scene
        '''
        # first find the scene
        scene_item = self._find_row({"context_type":"scene", "context":scene_id})

        # return the scene_item
        if shot_id==None:
            return scene_item

        # then find the shot
        if scene_item and shot_id:
            shot_item = self._find_row({"context_type"   :"shot",
                                       "context"        :shot_id},
                                        parent_item=scene_item[0])

            if shot_item:
                return shot_item[0]


    #    return the data hash contained in the given row
    def _get_row_data(self, tree_item, data_list=None):
        data_hash = {}
        if data_list==None:
            data_list = self._header_names

        for name in data_list:
            if name in self._header_names:
                column = self._header_names.index(name)

                # cast the data, for now, cast to string
                data_hash[name] = str( tree_item.text( column ) )

        return data_hash

    def _set_row_data(self, tree_item, data_hash):
        for name in data_hash.keys():
            value = data_hash[name]
            index = self._header_names.index(name)
            tree_item.setText(index, value)

    #    return the treeWidgetItem
    def _get_cell_widget(self, row_index_list, column_index, return_treeWidgetItem = False, return_root_item = False):
        '''
        Given the row index and column, index, return the cell widget it contains
        Note: later it would nice for the row index to be handle a hierachy list'


        * added return_treeWidgetItem / return_root_item flag, force return the parent treeWidget (row item so to speak)
        '''
        target_item = self.tree.invisibleRootItem()

        for row_index in row_index_list:
            target_item = target_item.child(row_index)
            if return_root_item == True:
                return target_item

        if return_treeWidgetItem == True:
            return target_item
        else:
            return self.tree.itemWidget(target_item, column_index)


    '''
    ============================================

    dealing with context selection

    ============================================
    '''
    #    check all the selected cell widgets
    def check_selected_cell(self):
        cell_widget_list = self._get_selected_cell_from_context()
        for i in range(len(cell_widget_list)):
            cell = cell_widget_list[i]
            self.logger.show_progress(len(cell_widget_list), i, "Adding sources...")

            if cell == None or len(cell._tank_address) == 0:
                continue

            cell.checkBox.setChecked(True)

    def clear_checked(self, clearAll = False, cell = None):
        '''
        there are two cases in which clear_checked is called:
        1. the user manually check/uncheck new or existing cells, causing the _checked_widget_list to change
        2. the user switches to a new tab(or either create anew or close the current), and the new tab may have totally
           different revisions, which in turn reflects on context view, so 'clearAll' flag should be set to True
        for the latter, the _checked_widget_list (along with the checkBoxes) is reset, then re-filled by whatever
        listed on mix master
        '''
        if cell:
            cell.flg_exec_check_actions = False
            cell.checkBox.setChecked(False)
            cell.flg_exec_check_actions = True
            cell.reset_default_version()
            return

        # the remove the check from review tool
        if clearAll == False and self._play_source=="context_view":
            # first remove the selection from mix master first
            include = []
            exclude = []
            selected_index = []

            for each_model_index in self.tree.selectionModel().selectedIndexes():
                selected_index.append(  (  each_model_index.parent().row(),
                                           each_model_index.row(),
                                           each_model_index.column() )  )
            for cell in self._checked_cell_widget:
                if (cell._parent_row, cell._row, cell._column) in selected_index:
                    include.append(cell)
                else:
                    exclude.append(cell)
            length = len(include)
            i = 0
            for cell in include:
                cell.flg_exec_check_actions = False
                cell.checkBox.setChecked(False)
                cell.flg_exec_check_actions = True
                cell.reset_default_version()

                self._sync_context_tree_to_mixmaster(cell, isChecked=True)

                self.logger.show_progress(length, i, "Removing sources from mix master...")
                i += 1

        elif clearAll == False and self._play_source=="playlist":
            self.get_active_mix_master_widget().removeSource()
            self._sync_mixmaster_to_context_tree()

        else:
            for cell in self._checked_cell_widget:
                cell.flg_exec_check_actions = False
                cell.checkBox.setChecked(False)
                cell.flg_exec_check_actions = True

            self._checked_cell_widget = []

    def _get_selected_cell_from_context(self):
        cell_widget_list = []
        model_selection = self.tree.selectionModel()
        all_selected_indexes = model_selection.selectedIndexes()

        for each_selected_index in all_selected_indexes:
            row_index_list = [ each_selected_index.parent().row(), each_selected_index.row()]
            column_index = each_selected_index.column() # from modelIndex's row/column search for the cell widget
            if self._header_names[column_index] in self._header_names:
                cell_widget = self._get_cell_widget(row_index_list, column_index)
                if cell_widget:
                    if cell_widget._tank_address:
                        cell_widget_list.append(cell_widget) # filter out the empty ones
                    else:   continue
                else:       continue
        return cell_widget_list

    def _clear_selection_for_checked_cells(self):
        cell_widget_list = []
        model_selection = self.tree.selectionModel()
        all_selected_indexes = model_selection.selectedIndexes()

        for each_selected_index in all_selected_indexes:
            row_index_list = [ each_selected_index.parent().row(), each_selected_index.row()]
            column_index = each_selected_index.column() # from modelIndex's row/column search for the cell widget
            cell_widget = self._get_cell_widget(row_index_list, column_index)

            if cell_widget.checkBox.isChecked():
                self.tree.setCurrentItem( cell_widget._context_row_item,
                                              column_index,
                                              QtGui.QItemSelectionModel.Deselect
                                              )


    def _select_whole_column(self, selected_column):
        if selected_column == 0:
            return
        selection_model = self.tree.selectionModel()
        selection_model.clear()
        root = self.tree.invisibleRootItem()
        selected_scene_indexes = [mi.row() for mi in self.tree_scene.selectionModel().selectedIndexes()]
        for scene_index in selected_scene_indexes:
            row_count = root.child(scene_index).childCount()
            if row_count == 0:
                continue
            else:
                for i in range(row_count):
                    self.tree.setCurrentItem(root.child(scene_index).child(i) , selected_column, QtGui.QItemSelectionModel.Select)

    def _sort_cell_widget(self, first, second):
        ''' read from _department_order list every time'''
        scene_order1, shot_order1 = first._source_order
        scene_order2, shot_order2 = second._source_order

        dept_name1  = first._context_widget._header_names[first._column]
        dept_order1 = first._context_widget._department_order.index(dept_name1)

        dept_name2  = second._context_widget._header_names[second._column]
        dept_order2 = second._context_widget._department_order.index(dept_name1)

        if scene_order1 < scene_order2:
            return -1
        elif scene_order1 > scene_order2:
            return 1
        else:
            if shot_order1 < shot_order2:
                return -1
            if shot_order1 > shot_order2:
                return 1
            else:
                # depends on 'department_order', which is user adjustable
                return cmp(dept_order1, dept_order2)


    def compare_with_method(self):
        self.play_with_method(use_compare=True)

    def play_with_method(self, use_compare=False):
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.logger.info(self._play_method)
            if self._play_method == "mode_playlist":
                self._play_mixMaster_playlist(use_compare)

            elif self._play_method == "mode_context_cut":
                self._play_context_without_handle(use_compare)

            elif self._play_method == "mode_context_source":
                self._play_context_with_handle(use_compare)

            elif self._play_method == "mode_smart_play":
                self._smart_play(use_compare)

        except:
            self.logger.error("Problem playing playlist. \n" + traceback.format_exc())

        QtGui.QApplication.restoreOverrideCursor()


    def _smart_play(self, use_compare):
        if self._play_source=="context_view":
            # get an override information for playlist.
            override_data = {}

            for i in range(self.get_active_mix_master_widget().tableView.getRowCount()):
                d = self.get_active_mix_master_widget().tableView.getShotData(i)

                if d["Revision"]:
                    aux_data = projectAwareness.determine_aux_data_from_tank_address_fast(d["Revision"])

                    override_data[(aux_data["scene"] , aux_data["shot"])] = d

            seq_data = self.get_active_mix_master_widget().get_sequence_data()
            self._play_context_selection(withHandle = False, use_compare=use_compare, override_data = override_data)

        else:
            self._play_mixMaster_playlist(use_compare)


    #    play with pre/post handle range
    def _play_context_with_handle(self, use_compare):
        self._play_context_selection(withHandle = True, use_compare=use_compare)

    #    play the actual cut length
    def _play_context_without_handle(self, use_compare):
        self._play_context_selection(withHandle = False, use_compare=use_compare)

    #    "direct play", generates a rv file
    def _play_context_selection(self, withHandle, use_compare, override_data={}):
        shot_data_list = self._create_shot_data_from_context_selection()

        if shot_data_list == None:
            self.logger.error("Nothing selected for playing, please select one or more shots from the context view.")
            return

        seq = rv_tools.edl.Sequence()

        for shot_data in shot_data_list:

            # if play with handles
            frame_min     = frame_in   = shot_data["source_start"]
            frame_max     = frame_out  = shot_data["source_end"]

            # if play with cut range
            if not withHandle and shot_data["aux_data"]["cut_range"]!=None:
                frame_in  = shot_data["aux_data"]["cut_range"]["sg_cut_start"]
                frame_out = shot_data["aux_data"]["cut_range"]["sg_cut_end"]

            # if there are override, use override data instead
            scene_id            = shot_data["aux_data"]["scene"]
            shot_id             = shot_data['aux_data']['shot']
            override_key        = (scene_id,shot_id)
            audio               = None
            use_clip_audio      = False
            audio_offset        = 0
            audio_frame_start   = -1
            stereo_pair         = None

            if override_key in override_data.keys():
                shot_overrides = override_data[override_key]
                frame_in  = shot_overrides["Start"]
                frame_out = shot_overrides["End"]

                # collect information from the overrides table
                audio               = shot_overrides["Audio"] if shot_overrides["Audio"] else None
                audio_offset        = shot_overrides['Audio Offset']
                audio_frame_start   = shot_overrides['Audio Start']

                use_clip_audio  = True if shot_overrides["Audio Override"] else False

                # only use the audio offset for movies
                if ( self._play_frame_sequences == 'source_img_seq' and audio_frame_start != None ):
                    audio_offset = audio_frame_start - frame_in
                    audio_frame_start = -1

                # grab the sinfo data from tank
                sinfo_path          = ''
                sinfo_data          = {}
                sinfo_address       = shot_overrides['Revision'].replace('Frames','SInfoFile').replace('Movie','SInfoFile')

                # populate the sinfo address to determine the sinfo file
                if ( sinfo_address ):
                    try:
                        sinfo_path = tank.find(sinfo_address).system.filesystem_location
                    except:
                        sinfo_path = ''

                    if ( sinfo_path and os.path.exists(sinfo_path) ):
                        sinfo_data = sinfo_utils.unpickle_dict(sinfo_path)

                # extract the stereo pair from the sinfo data
                if ( sinfo_data ):
                    stereo_pair = sinfo_data.get('stereo_pair')

            c = seq.add_clip(
                            left                = shot_data["file_full_path"],
                            frame_in            = frame_in,
                            frame_out           = frame_out,
                            frame_min           = frame_min,
                            frame_max           = frame_max,
                            audio               = audio,
                            use_clip_audio      = use_clip_audio,
                            audio_offset        = audio_offset,
                            audio_frame_start   = audio_frame_start,
                            stereo_pair         = stereo_pair
                         )

            c.set_meta_data ( "scene_shot", shot_data["aux_data"]["shot"],
                              label="Shot",  hidden=False )
            c.set_meta_data ( "department", shot_data["aux_data"]["department"],    label="Department",       hidden=False )
            c.set_meta_data ( "scene_shot_index", shot_data["aux_data"]["cut_order"],                         hidden=True )
            c.set_meta_data ( "tank_rev_obj", str(shot_data["aux_data"]["tank_rev_obj"]),
                              label="Tank Revision",  hidden=False )


        rvs = rv_tools.get_rv_session(new_session=self.checkBox_rv_singleton.isChecked())

        if use_compare:
            seq_list = seq.group_by_department(
                                               department_meta_id = "department",
                                               scene_shot_meta_id = "scene_shot",
                                               row_order_meta_id  = "scene_shot_index",
                                               department_order   = self._department_order
                                               )

            rvs.compare(seq_list, mode=self._play_compare_mode)
        else:
            rvs.open(seq)

        rvs.eval("resizeFit()")



    #    play the list from mix master widget
    #    wrapping playall()
    def _play_mixMaster_playlist(self, use_compare):
        flg_movie_source = True if self._play_frame_sequences=="source_movie" else False

        if self.get_active_mix_master_widget().tableView.model().rowCount()==0:
            self.logger.error("Playlist is empty.")
            return

        self.get_active_mix_master_widget().play_all(flg_movie_source=flg_movie_source,
                                                     compare_mode=self._play_compare_mode if use_compare else None,
                                                     department_order=self._department_order,
                                                     rv_session = rv_tools.get_rv_session(new_session=self.checkBox_rv_singleton.isChecked()))


    def _create_shot_data_from_context_selection(self):
        cell_widget_list = self._get_selected_cell_from_context()
        if cell_widget_list == None or len(cell_widget_list) == 0: return None
        elif cell_widget_list[0] == None:                          return None

        cell_widget_list.sort(self._sort_cell_widget) # sort by cell's _source_order
        shot_data_list  = [] # example: [{...}, {...}, {...}......]
        i = 0
        for cell in cell_widget_list:
            data = self._get_shot_data_from_cell(cell)
            shot_data_list.extend(data)

            self.logger.show_progress(len(cell_widget_list), i, "Generating RV file for selected shots...")
            i += 1

        return shot_data_list

    def _get_shot_data_from_cell(self, cell):
        '''
        input:  a cellWidget object
        return: shot_data list, example : [(ver, sg_cut_info, frame_info), (...), (...).......]
        '''
        if len(cell._tank_address) == 0: # meaning the selection is grey-out
            return None

        shot_data=[]

        for tank_rev_address in cell._tank_address.split("|@|"): # this by default is the frame address!!
            # get the source range from the movie

            movie_obj       = tank.find( tank_rev_address.replace('Frames(', 'Movie(') )
            frame_obj       = tank.find( tank_rev_address.replace('Movie(', 'Frames(') )
            source_obj      = movie_obj if  self._play_frame_sequences=="source_movie" else frame_obj

            if TANK_VFS_SUCKS:
                frame_full_path = frame_obj.system.filesystem_location
                movie_full_path = movie_obj.system.filesystem_location
            else:
                frame_full_path = frame_obj.system.vfs_full_paths[0]
                movie_full_path = movie_obj.system.vfs_full_paths[0]

            fpath, fname, source_start, source_end = get_frame_sequence_source_fast(frame_full_path)
            full_path = movie_full_path if self._play_frame_sequences=="source_movie" else os.path.join(fpath, fname)

            shot_data.append( {
                    "aux_data":       projectAwareness.determine_aux_data_from_tank_address(source_obj, include_shotgun_data=True),
                    "file_full_path": full_path,
                    "source_start":   source_start,
                    "source_end":     source_end,
                    })

        return shot_data


    def _sync_play_button_state(self):
        self.compare_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.button_clear_check.setEnabled(True)

        if self._play_method=="mode_smart_play" and not self._get_selected_cell_from_context() and not self.get_active_mix_master_widget().tableView.getSelectedRow():
            self.compare_button.setEnabled(False)
            self.play_button.setEnabled(False)

        self.button_check_select.setEnabled(False)
        if self._get_selected_cell_from_context() and self._play_source=="context_view":
            self.button_check_select.setEnabled(True)

        if not self._get_selected_cell_from_context() and not self.get_active_mix_master_widget().tableView.getSelectedRow():
            self.button_clear_check.setEnabled(False)


    def _handle_save_play_option(self, action):
        if action.isChecked():
            action_data = str(action.data().toString())

            if action_data in [ m[0] for m in self._play_frame_sequences_types ]:
                self._play_frame_sequences = action_data

            if action_data in [ m[0] for m in self._play_method_types ]:
                self._play_method = action_data

            if action_data in [ m[0] for m in self._play_source_locations ]:
                self._play_source_location = action_data
                rv_tools.LOCALIZE_SESSION = action_data == 'local_source'

            if action_data in [ m[0] for m in self._compare_mode_types ]:
                self._play_compare_mode = action_data
                self.compare_button.setIcon([ m[2] for m in self._compare_mode_types if m[0] == action_data][0])

            self._sync_play_button_state()


    def _handle_show_play_option(self):

        for action in self.play_option_menu.actions():
            action_data = str(action.data().toString())

            if action_data in [self._play_frame_sequences, self._play_method, self._play_compare_mode, self._play_source_location]:
                action.setChecked(True)


    '''
    ==============================================================

    processing mix master widget: selection ; tabs creating/deleting

    ==============================================================
    '''

    #    the actual tab creating method
    #     returns the new tab index
    def create_new_tab(self, tab_name = None, tab_index= None, tab_tool_tip="", tab_icon=None):
        if not tab_name:
            tab_name = "Playlist %02d" % ( self._next_tab_id )

        self._next_tab_id +=1

        mix_master_widget = MixMasterWidget(self.playlist_tab_panel, prefSaver=None, thumbTmpRoot=THUMB_TEMP_ROOT, context_project = self._project, context_widget=self)

        palette = mix_master_widget.palette()
        refPalette = self.tree.palette()

        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, refPalette.base())
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, refPalette.alternateBase())
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, refPalette.base())
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, refPalette.alternateBase())

        mix_master_widget.tableView.setPalette( palette )

        if not tab_index: tab_index = self.playlist_tab_panel.count()-1

        new_index = self.playlist_tab_panel.insertTab(tab_index, mix_master_widget, tab_name)

        self.playlist_tab_panel.setCurrentIndex(new_index)
        self.playlist_tab_panel.setTabToolTip(new_index, tab_tool_tip)
        if tab_icon: self.playlist_tab_panel.setTabIcon(tab_index, tab_icon)

        self.connect(
                     mix_master_widget.tableView.selectionModel(),
                     QtCore.SIGNAL("selectionChanged (const QItemSelection&, const QItemSelection&)"),
                     self._handle_mix_master_selection_change
                     )

        context_button_menu = QtGui.QMenu(mix_master_widget.load_saver_button)
        context_button_menu.addAction(
                                      QtGui.QIcon(self.shotgun_icon),
                                      "Open Playlist From Shotgun..",
                                      self._handle_load_shotgun_playlist
                                      )

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("page_white_get.png")),
                                      "Open Playlist (or Final Cut Xml) from File..",
                                      self._handle_import_playlist
                                      )

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("page_save.png")),
                                      "Save Playlist to File..",
                                      self._handle_save_playlist
                                      )

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("font.png")),
                                      "Rename Playlist..",
                                      self._handle_rename_playlist
                                      )

        context_button_menu.addSeparator ()

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("desktop_empty.png")),
                                      "Open Review Session..",
                                      self._handle_open_review_session
                                      )

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("desktop.png")),
                                      "Save Review Session..",
                                      self._handle_save_review_session
                                      )

        context_button_menu.addSeparator ()

#        context_button_menu.addAction(
#                                      QtGui.QIcon(resource.getImage("desktop.png")),
#                                      "Open meeting..",
#                                      self._handle_open_meeting
#                                      )


#        context_button_menu.addAction(
#                                      QtGui.QIcon(resource.getImage("desktop.png")),
#                                      "Save meeting..",
#                                      self._handle_save_meeting
#                                      )


        context_button_menu.addSeparator ()

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("sound_add.png")),
                                      "Override selection with published audio",
                                      mix_master_widget._handle_override_audio_for_selection
                                      )

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("sound_sub.png")),
                                      "Restore selection to use embedded audio (if any)",
                                      mix_master_widget._handle_override_audio_for_selection_remove
                                      )

        context_button_menu.addAction(
                                     QtGui.QIcon(resource.getImage('switch_department.png')),
                                     'Switch departments for selection...',
                                     mix_master_widget.rclickSwitchDepartmentSel
                                     )

        context_button_menu.addAction(
                                     QtGui.QIcon(resource.getImage('update_version.png')),
                                     'Update selection to latest version',
                                     mix_master_widget.rclickUpdateVersionSel
                                     )

        context_button_menu.addSeparator ()

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("add.png")),
                                      "Add external file source...",
                                      self._handle_add_source_to_playlist
                                      )

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage("cog.png")),
                                      "Add technical/creative renders...",
                                      self._handle_review_mapping
                                      )

        context_button_menu.addSeparator()

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage('contact_sheet.png')),
                                      'Generate Contact Sheet...',
                                      self._handle_create_contact_sheet )

        context_button_menu.addSeparator()

        context_button_menu.addAction(
                                      QtGui.QIcon(resource.getImage('statistics.png')),
                                      'Compare breakdown/package versions',
                                      self.start_bkd_diag )

        self.connect(mix_master_widget, QtCore.SIGNAL("render_version_updated"), self._handle_playlist_version_update)

        mix_master_widget.load_saver_button.setMenu(context_button_menu)

        return new_index


    def _handle_playlist_version_update(self, context_shot, ori_dept, ori_ver, new_ver, new_dept=None):
        """
        User updates the version or the department in the playlist,
        Now need to sync to the context tree to be same data
        """
        shot_item = self.find_tree_item_from_context(scene_id = context_shot.scene.name, shot_id=context_shot.name)

        ori_dept_cell = self.tree.itemWidget( shot_item, self._header_names.index(ori_dept) )
        if new_dept:
            new_dept_cell = self.tree.itemWidget( shot_item, self._header_names.index(new_dept) )
        else:
            new_dept_cell = ori_dept_cell

        # first remove from original department cell
        if ( ori_dept_cell ):
            rev_str = str(ori_dept_cell.name_label.text())
        else:
            rev_str = ''

        if rev_str:
            rev_list = rev_str.split(", ")
            if ( ori_ver in rev_list ):
                rev_list.remove(ori_ver)

            ori_dept_cell.name_label.setText( ", ".join(rev_list) )

            if len(rev_list)==0 and ori_dept_cell!=new_dept_cell:
                self.ignore_context_tree_check_events = True
                ori_dept_cell.checkBox.setChecked(False)
                self.ignore_context_tree_check_events = False

        # make sure we have a new cell to work with
        if ( new_dept_cell ):
            # now add the new version to new department cell
            rev_str = str(new_dept_cell.name_label.text())
            if rev_str:
                rev_list = rev_str.split(", ")
                rev_list = list(set(rev_list))
            else:
                rev_list=[]

            if new_dept_cell.checkBox.isChecked(): # only need to add if already checked, else by check it automatically add new
                rev_list.append(new_ver)
                rev_list.sort()
                rev_list.reverse()
            else: # new_dept_cell.checkBox.isChecked():
                self.ignore_context_tree_check_events = True
                new_dept_cell.checkBox.setChecked(True)
                self.ignore_context_tree_check_events = False

            new_dept_cell.name_label.setText( ", ".join(rev_list) )

            render_ver = context_shot.get_render_shotgun(new_dept if new_dept else ori_dept, rev_list[0])

            if render_ver:
                new_dept_cell.set_status_icon(render_ver["sg_data"]["sg_status_1"])

            self._handle_mix_master_selection_change()

    def _handle_create_contact_sheet( self ):
        """
            :remarks    prompts the user to create a new contact sheet based
                        on the currently selected playlist
        """

        QtGui.QApplication.setOverrideCursor( QtCore.Qt.WaitCursor )

        self.logger.info('Importing contactSheet module...')

        # make sure to set the context for the contact sheet to be the review tool
        from contactSheet.generator             import ContactSheetGenerator
        from contactSheet.gui.generatorwindow   import ContactSheetGeneratorWindow

        self.logger.info('Restoring previous settings...')

        # restore the last generator based on user preferences
        generator = ContactSheetGenerator.restoreLast('Review Tool')

        self.logger.info('Saving out playlist file...')

        # save out a temp RV file to use
        playlist = os.path.normpath(os.path.expandvars('$DRD_TEMP/rv_playlist.rv'))
        self.get_active_mix_master_widget().save_file(playlist)

        self.logger.info('Loading playlist for Contact Sheet')
        # set it as the generator's playlist
        generator.setPlaylist(playlist)

        # edit the generator
        ContactSheetGeneratorWindow.edit(generator, parent = self)

        QtGui.QApplication.restoreOverrideCursor()

    def _handle_add_source_to_playlist(self):
        mm = self.get_active_mix_master_widget()
        mm.txtEditSourcePath.setText(self.user_browsed_path)
        mm.handleAddSource()
        mm.generate_dynamic_data()
        self._sync_mixmaster_to_context_tree()

    def _handle_review_mapping(self):
        # collect the shot data
        shot_data_list = self._create_shot_data_from_context_selection()

        if ( not shot_data_list ):
            QtGui.QMessageBox.critical( self, 'No Context Selected', 'There is no context selected to create the render mapping for.' )
            return False

        # create the review mapper
        from .reviewmapperdialog import ReviewMapperDialog
        reviews, accepted = ReviewMapperDialog.collectPairing(shot_data_list,self)
        if ( accepted ):
            # get the selected cell widgets
            cell_widget_list    = self._get_selected_cell_from_context()
            cell_widget_data    = {}

            # map the resulting data to the cell widgets
            for cell_widget in cell_widget_list:
                data        = self._get_shot_data_from_cell(cell_widget)
                for shot_data in data:
                    shot_addr   = str(shot_data['aux_data']['tank_asset_rev_obj'])
                    for base, additional in reviews.items():
                        label, address = base.split('|')
                        if ( shot_addr == address ):
                            cell_widget_data.setdefault(cell_widget,{'label_list':[],'address_list':[]})
                            cell_data = cell_widget_data[cell_widget]
                            cell_data['label_list'].append(label)
                            cell_data['address_list'].append(address)

                            for rdata in additional:
                                rlabel,raddress = rdata.split('|')
                                cell_data['label_list'].append(rlabel)
                                cell_data['address_list'].append(raddress)

            # update the cell widget data
            for cell_widget, data in cell_widget_data.items():
                cell_widget.name_label.setText( ", ".join(data["label_list"]) )
                cell_widget._tank_address = "|@|".join(data["address_list"])

                self._sync_context_tree_to_mixmaster(cell_widget, isChecked=True)

    def _handle_rename_playlist(self):
        tab_index = self.playlist_tab_panel.currentIndex()
        mm = self.get_active_mix_master_widget()

        if mm.shotgun_playlist_id!=None:
            QtGui.QMessageBox.information(self, self.tr("Shotgun playlist"),
                                          self.tr("Shotgun playlist can not be renamed."))
            return

        msg = "Enter name for playlist:"
        name, ok = QtGui.QInputDialog.getText(self, self.tr("Playlist name"),
                    self.tr(msg), QtGui.QLineEdit.Normal,
                    self.playlist_tab_panel.tabText(tab_index))

        if ok and str(name):
            self.playlist_tab_panel.setTabText(tab_index, str(name))

    def _handle_open_review_session(self):
        # prompt for a file if none is given
        tab_index = self.playlist_tab_panel.currentIndex()
        fileName = QtGui.QFileDialog.getOpenFileName(self,
                                     self.tr("Import review session"),
                                     self.user_browsed_path,
                                     self.tr("Review Session (*.zip);;All files (*.*)")
                                     )
        if fileName.isEmpty():
            return

        if not(os.path.isfile(fileName)):
            print "Warning: %s does not exist." % fileName
            return

        fileName = str(fileName)

        # remove all existing tabs:
        self._ignore_tab_change = True
        for tab_index in range(self.playlist_tab_panel.count()-1):
            self.playlist_tab_panel.removeTab(tab_index)

        # create the tmp directory if it doesn't alrady exist
        tmp_dir = os.path.join("/tmp", os.path.split(fileName)[1].split(".")[0])
        if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)

        import zipfile
        z = zipfile.ZipFile(fileName, 'r')

        self.set_window_title(fileName)

        tab_index = 0
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        for playlist in z.namelist():
            tmp_playlist_file = os.path.join(tmp_dir, playlist)

            content = z.read(playlist)
            f = open(tmp_playlist_file, 'w')
            f.write(content)
            f.close()

            pl_filename = os.path.split(str(playlist))[-1]
            sg_id = None
            if "shotgunId" in pl_filename:
                sg_id, pl_label= re.search("shotgunId([0-9]+)_([\w_\s]+)\.[\w]{2,4}$", pl_filename).groups()

            else:
                pl_label = re.search("[0-9]+_([\w_\s]+)\.[\w]{2,4}$", pl_filename).groups()[0]

            new_index = self.create_new_tab(str(pl_label), tab_index, tab_tool_tip = playlist)


            mm = self.playlist_tab_panel.widget(new_index)
            if sg_id:
                mm.shotgun_playlist_id = int(sg_id)
                self.playlist_tab_panel.setTabIcon(tab_index, self.shotgun_icon)

            tab_index +=1

            # clear the logger
            self.logger.reset()

            # import the playlist
            self.logger.info( 'Importing playlist %s...\n--------------------------------------' % tmp_playlist_file )
            print "Importing playlist %s..." % tmp_playlist_file

            # known errors will be caught and logged
            try:
                mm.import_playlist(filename=str(tmp_playlist_file))
                # do some sync
                self._sync_mixmaster_to_context_tree()

            # any pythonic/unknown errors will be caught and logged here
            except:
                import traceback
                self.logger.error('[UNKNOWN ERROR]:\n' + traceback.format_exc())

            # look for any errors that occurred
            error_count = self.logger.message_count('error')
            if ( error_count ):
                self.logger.error('%i errors occurred during import.' % error_count)

        z.close()
        self._ignore_tab_change = False
        self._sync_mixmaster_to_context_tree()
        QtGui.QApplication.restoreOverrideCursor()


    def _handle_save_review_session(self):
        import zipfile
        fileName = QtGui.QFileDialog.getSaveFileName(self,
                                         self.tr("Save Playlist"),
                                         self.user_browsed_path,
                                         self.tr("RV Files (*.zip)"))
        fileName = str(fileName).strip()

        if not(fileName.endswith(".zip")):
            fileName += ".zip"

        if not fileName:
            return

        zf = zipfile.ZipFile(fileName, "w")

        # create the tmp directory if it doesn't alrady exist
        tmp_dir = os.path.join("/tmp", os.path.split(fileName)[1].split(".")[0])
        if not os.path.exists(tmp_dir): os.mkdir(tmp_dir)

        for tab_index in range(self.playlist_tab_panel.count()-1):
            self.playlist_tab_panel.setCurrentIndex(tab_index)
            mm = self.get_active_mix_master_widget()

            plName = "%s%s_%s.rv" % (tab_index,
                                     "_shotgunId%s" % ( mm.shotgun_playlist_id ) if mm.shotgun_playlist_id else "",
                                     str(self.playlist_tab_panel.tabText(tab_index))
                                     )

            plPath = os.path.join(tmp_dir, plName)
            self.logger.info("Saving %s..." % plPath)
            saved_file = self.get_active_mix_master_widget().save_file(plPath)

            if saved_file:
                zf.write(plPath, plName)

        # clean up the tmp directory and close file
        zf.close()

        import shutil
        shutil.rmtree(tmp_dir)


    def _handle_save_playlist(self):
        fileName = QtGui.QFileDialog.getSaveFileName(self,
                                         self.tr("Save Playlist"),
                                         self.user_browsed_path,
                                         self.tr("RV Files (*.rv)"))
        fileName = str(fileName).strip()

        if not(fileName.endswith(".rv")):
            fileName += ".rv"

        if fileName!="":
            self.get_active_mix_master_widget().save_file(fileName)
            self.user_browsed_path = os.path.split(str(fileName))[0]

    def _handle_import_playlist(self):
        # prompt for a file if none is given
        tab_index = self.playlist_tab_panel.currentIndex()
        fileName = QtGui.QFileDialog.getOpenFileName(self,
                                     self.tr("Import playlist"),
                                     self.user_browsed_path,
                                     self.tr("RV and Final Cut Xml files (*.rv *.xml);;All files (*.*)")
                                     )
        if fileName.isEmpty():
            return
        if not(os.path.isfile(fileName)):
            print "Warning: %s does not exist." % fileName
            return

        self._ignore_tab_change = True

        new_index = self.create_new_tab(os.path.split(str(fileName))[-1], tab_index + 1, tab_tool_tip = fileName)
        mm = self.playlist_tab_panel.widget(new_index)
        self._ignore_tab_change = False

        self.user_browsed_path = os.path.split(str(fileName))[0]

        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # clear the logger
        self.logger.reset()

        # import the playlist
        self.logger.info( 'Importing playlist %s...\n--------------------------------------' % fileName )

        print "Importing playlist %s..." % fileName

        # known errors will be caught and logged
        try:
            mm.import_playlist(filename=str(fileName))
            # do some sync
            self._sync_mixmaster_to_context_tree()

        # any pythonic/unknown errors will be caught and logged here
        except:
            import traceback
            self.logger.error('[UNKNOWN ERROR]:\n' + traceback.format_exc())

        # look for any errors that occurred
        error_count = self.logger.message_count('error')
        if ( error_count ):
            self.logger.error('%i errors occurred during import.' % error_count)

        QtGui.QApplication.restoreOverrideCursor()



    def _handle_tab_change(self, tab_index):
        # the last one clicked, ceate a new tab
        if self._ignore_tab_change:
            return

        self._ignore_tab_change = True

        if tab_index==self.playlist_tab_panel.tabBar().count()-1:
            self.create_new_tab(tab_index=tab_index)

            self._sync_mixmaster_to_context_tree()

            if self._play_source=="playlist":
                self._handle_mix_master_selection_change()
        else:
            self._sync_mixmaster_to_context_tree()

            if self._play_source=="context_view":
                self._handle_tree_cell_selection_change()
            else:
                self._handle_mix_master_selection_change()

        self._ignore_tab_change = False


    def close_tab(self, tab_index):
        if tab_index!=self.playlist_tab_panel.tabBar().count()-1 and self.playlist_tab_panel.tabBar().count()> 2:
            if self.playlist_tab_panel.widget( tab_index ).top_row_count()!=0:
                reply = QtGui.QMessageBox.question(self,
                                "Close Playlist?",
                                "Close non-empty playlist \"%s\"?" % self.playlist_tab_panel.tabText( tab_index ),
                               QtGui.QMessageBox.Ok,
                               QtGui.QMessageBox.Cancel)

                if reply == QtGui.QMessageBox.Cancel:
                    return

            self._ignore_tab_change = True
            self.playlist_tab_panel.removeTab(tab_index)
            self._ignore_tab_change = False

            # ensure the new tab is not selected
            if tab_index == self.playlist_tab_panel.tabBar().count()-1:
                self.playlist_tab_panel.setCurrentIndex(tab_index-1)
            self._sync_mixmaster_to_context_tree()


    #    return the current active mix master widget
    def get_active_mix_master_widget(self):
        return self.playlist_tab_panel.currentWidget()


    '''
    ============================================

    context-mixMaster selection synchronization

    ============================================
    '''
    #    replaced _sync_cell_widgets_from_mix_master_selection
    def _handle_mix_master_selection_change(self, new_selection=None, un_selection=None, load_scene_shot_if_required=True):
        '''
        this method ONLY handle the selection-change, it does not care about checkBox

        load_scene_shot_if_required means if the playlist shot selected requires load the tree first, then do it.

        if un_selection and  new_selection not specify it will sync the entire playlist selection to the context view
        '''
        if self.ignore_mixmaster_selection_events: return

        self.ignore_context_tree_selection_events = True

        mm = self.get_active_mix_master_widget()

        flg_add_selection = True
        self.tree.selectionModel().clearSelection()


        update_select_index = mm.tableView.getSelectedRow() # a list of tuples: (parent_row_index, child_row_index)

        shot_item = None
        shot_data_list = []

        for selected_index in update_select_index:
            shot_data = mm.tableView.getShotData(selected_index[0], selected_index[1]) if len(selected_index) == 2 else mm.tableView.getShotData(selected_index[0])
            full_vfs_path = os.path.join(shot_data["Source Path"], shot_data["Source Name"])

            aux_data = shot_data["_Aux_Data"] #projectAwareness.determine_aux_data_from_vfs_path(full_vfs_path)

            if aux_data:
                context_type, scene_id, shot_id, department = aux_data.split(",")
                if not scene_id in self._project.list_scene_names():
                    continue

                scene_index = self._project.list_scene_names().index(scene_id)

                if load_scene_shot_if_required:
                    self.tree_scene.setItemSelected(  self.tree_scene.invisibleRootItem().child(scene_index), True )

                self.tree_scene.scrollToItem(self.tree_scene.invisibleRootItem().child(scene_index), QtGui.QAbstractItemView.PositionAtCenter)

                shot_item = self.find_tree_item_from_context(scene_id=scene_id, shot_id=shot_id)
                if shot_item:
                    self.tree.setCurrentItem( shot_item,
                                              self._header_names.index( department ),
                                              QtGui.QItemSelectionModel.Select if flg_add_selection else QtGui.QItemSelectionModel.Deselect
                                              )
                else:
                    continue

        self.toggle_play_source(play_source="playlist")
        self.ignore_context_tree_selection_events = False


    '''
    ============================================

    shotgun playlist loader dialogue

    ============================================
    '''
    def populate_mix_master_from_shotgun(self, playlist_id=None, clip_id_list=None):
        '''
        Given a shotgun playlist id, query tank for version in playlist and populate mix master.
        '''
        api = shotgun_session().api()
        version_query_fields=["code", 'sg_rv_preview_link', 'sg_tank_address']

        if playlist_id!=None:
            self.logger.info("loading playlist from shotgun...")
            result_pl = api.find_one("Playlist", [["id","is",playlist_id]],["code","versions"])
            playlist_name = result_pl["code"]
            clip_id_list = [ v["id"] for v in result_pl["versions"] ]
            search_criteria = [ ["id","is", id] for id in clip_id_list ]
            result = api.find('Version', search_criteria, filter_operator='any',fields=version_query_fields)
            result.sort(lambda x, y: cmp( clip_id_list.index(x["id"]), clip_id_list.index(y["id"])) )

        else:
            self.logger.info("loading clips from shotgun...")
            search_criteria = [ ["id","is", id] for id in clip_id_list ]
            result = api.find('Version', search_criteria, filter_operator='any',fields=version_query_fields)
            result.sort(lambda x, y: cmp( clip_id_list.index(x["id"]), clip_id_list.index(y["id"])) )
            playlist_name = None

        self.create_new_playlist_with_clip_data(result, playlist_name, shotgun_id=playlist_id)


    def create_new_playlist_with_clip_data(self, clipdata, playlist_name=None, shotgun_id=None):
        '''
        Create a new playlist with the clip data from shotgun.
        If playlist name is specified, a new playlist will be created, otherwise just put to the first playlist
        '''
        result = clipdata
        # create a new mix master tab
        self._ignore_tab_change = True

        if playlist_name:
            new_index = self.create_new_tab(playlist_name,
                                            tab_icon = self.shotgun_icon if self.shotgun_icon else None)

            mm = self.playlist_tab_panel.widget(new_index)
        else:
            mm = self.playlist_tab_panel.widget(0)

        if shotgun_id:
            mm.shotgun_playlist_id = shotgun_id

        self._ignore_tab_change = False

        #  now populate the data
        rg_search_file_path = re.compile("file_path([\w\s_/.]+)")

        for i in range(len(result)):

            render_data = result[i]
            self.logger.show_progress(range=len(result), value=i, msg="Loading renders...")

            # check if there is tank address
            tank_address = render_data['sg_tank_address']

            if not tank_address:
                self.logger.warn("Can not find tank address for shotgun record " % result[i] )
                continue

            # if there is tank address in the version, then use that
            # other parse the address out of the rv link
            if tank_address:
                aux_data = projectAwareness.determine_aux_data_from_tank_address(tank_address)
            else:
                if render_data['sg_rv_preview_link']:
                    file_path = rg_search_file_path.search(render_data['sg_rv_preview_link']).groups()[0].strip()
                    aux_data = projectAwareness.determine_aux_data_from_vfs_path(file_path)
                    if aux_data: tank_address = str(aux_data["tank_rev_obj"])
                else:
                    continue

            if aux_data == None:
                print "Failed to get determine aux data from shotgun record", render_data
                continue

            tank_address = tank_address.replace("Movie", "Frames")

            # now get the cut data
            cut_info = None
            if self._project.get_scene(aux_data["scene"]) and self._project.get_scene(aux_data["scene"]).get_shot(aux_data["shot"], no_exception=True):
                cut_info = self._project.get_scene(aux_data["scene"]).get_shot(aux_data["shot"]).get_cut_info()

            tank_rev_obj = tank.find(tank_address)

            if TANK_VFS_SUCKS:
                full_path = tank_rev_obj.system.filesystem_location
            else:
                full_path = tank_rev_obj.system.vfs_full_paths[0]

            frame_path, frame_name, start, end = get_frame_sequence_source_fast(full_path)

            shot_data_hash = {
                            'Source Name'   :   frame_name,
                            'Source Path'   :   frame_path,
                            'Start'         :   cut_info['sg_cut_start'] if cut_info else start,
                            'End'           :   cut_info['sg_cut_end'] if cut_info else end,
                            'Min'           :   start,
                            'Max'           :   end,
                            }

            mm.tableView.addShot(shot_data_hash, i)

            # generate for all the data from shotgun
            mm.generate_dynamic_data(row_index_list=[i])

        self._sync_mixmaster_to_context_tree()
        mm._select_all_sources()


    #    pop up a shotgun-append-playlist dialog
    def _handle_load_shotgun_playlist(self):
        self.ap = ShotgunApWidget(self.pref_saver.getWidgetGroup("shotgun_playlist_loader"))

        if self.ap.exec_() and self.ap._shotgun_id!=None: #shotgun_playlist_id == None:
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self.populate_mix_master_from_shotgun(self.ap._shotgun_id)
            except:
                self.logger.hide_progress()
                self.logger.error("Error loading playlist: \n" + traceback.format_exc())

            QtGui.QApplication.restoreOverrideCursor()


    '''
    =========================================

           department filter dialog

    =========================================
    '''
    def _department_filter_loader(self):
        display_department_indexes = []

        self.deptFilter =  DepartmentFilterWidget(self)
        self.deptFilter.reset_order_button.hide()

        for d in self._department_order:
            item = QtGui.QListWidgetItem(self.deptFilter.list)
            item.setText( DEPARMENT_LABELS[d] if DEPARMENT_LABELS.has_key(d) else d )
            item.setData(QtCore.Qt.ToolTipRole, QtCore.QVariant(d))
            item.setCheckState(QtCore.Qt.Checked if d in self._shown_departments else QtCore.Qt.Unchecked)

        self.deptFilter.show()
        self.connect(
                     self.deptFilter.buttonBox,
                     QtCore.SIGNAL("accepted()"),
                     self._department_filter_accepted
                     )


    def _department_filter_accepted(self):
        shown_departments   = []
        department_order    = []
        for i in range(self.deptFilter.list.count()):
            item = self.deptFilter.list.item(i)
            dept = str(item.data(QtCore.Qt.ToolTipRole).toString())
            department_order.append(dept)
            if item.checkState() == QtCore.Qt.Checked:
                shown_departments.append(dept)

        print "Updated shown department: ", shown_departments
        print "Updated department order: ", department_order

        if set(self._shown_departments)!=set(shown_departments):
            self._handle_refresh_tool(prompt_first=False)

        self._shown_departments = shown_departments
        self._department_order  = department_order

        self._refresh_department_column_state()




    def _refresh_department_column_state(self):
        """
        Sync the the gui to the state as dictated by:
        self._shown_departments
        self._department_order
        """
        department_first_index = self._header_names.index(DEPARTMENTS[0])

        for i in range(len(self._department_order)):
            dept = self._department_order[i]

            self.tree.header().swapSections(
                                     self.tree.header().visualIndex(self._header_names.index(dept)),
                                     department_first_index + i
                                    )

            if dept in self._shown_departments:
                self.tree.showColumn( self._header_names.index(dept) )
            else:
                self.tree.hideColumn( self._header_names.index(dept) )

    '''
    ============================================

    status preferences

    ============================================
    '''
    def refresh_contents( self ):
        QtGui.QApplication.setOverrideCursor( QtCore.Qt.WaitCursor )

        w = self.playlist_tab_panel.currentWidget()
        w.setUpdatesEnabled(False)

        # reload the tree items
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)

            # reload pre-loaded items
            if ( not (item.childCount() == 1 and item.child(0).text(0) == 'dummy') ):
                self._handle_expand_scene(item,force = True)

        w.setUpdatesEnabled(True)

        QtGui.QApplication.restoreOverrideCursor()

    def refresh_sort_method( self, action ):
        ContextSession.SORT_METHOD = str(action.data().toString())
        self.refresh_contents()

    def set_sort_method( self, method ):
        ContextSession.SORT_METHOD = method

        # update the sorting menu to the latest sort method options
        for action in self.status_filter_button.menu().findChildren(QtGui.QAction):
            data = str(action.data().toString())
            if ( data.startswith('sort_by') ):
                action.setChecked( data == ContextSession.SORT_METHOD )

    def show_shot_status_filter( self ):
        from .statusprefsdialog import StatusPrefsDialog

        tip = 'Filter visible shots by checking off the statuses you want to see loaded.'

        # let the user edit their prefs then reload
        if ( StatusPrefsDialog.edit(self,SHOT_STATUS_HASH,self._shotStatusPrefs,draggable=False,tip=tip) ):
            self.refresh_contents()

    def show_status_filter( self ):
        from .statusprefsdialog import StatusPrefsDialog

        # let the user edit their prefs then reload
        if ( StatusPrefsDialog.edit(self, ContextCell.render_status_hash, self._statusPrefs ) ):
            ContextSession.SORT_STATUS_ORDER = self._statusPrefs.values('order')
            self.refresh_contents()

    def start_bkd_diag(self):
        '''
        get all playlist tabs except for the default 'new'
        query for the frame revision address from all the shots in the mixMaster widgets
        and get the sinfo file system location from there, then run breakdown_diagnosis
        '''
        collection = []
        wcount = self.playlist_tab_panel.count()
        for i in range(wcount):
            tab_name = str(self.playlist_tab_panel.tabText(i))
            if tab_name == 'New':
                continue
            iw = self.playlist_tab_panel.widget(i)
            rcount = iw.top_row_count()
            tab_data = []
            for r_number in range(rcount):
                data = iw.get_top_row_data(r_number)
                # if the Revision entry does not exist in the data dictionary
                # it'll end up with "Scene(9999)" which will raise a standard
                # tank.common.errors.TankNotFound
                sinfo_address = data.get('Revision', 'Scene(9999)').replace('Frames', 'SInfoFile')
                updated_date = data.get('Updated', '--:--:-- ----')
                version = data.get('Version', '---')
                title = data.get('_Aux_Data', 'Unknown clip')
                display_title = "%s   ver: %s   updated_at: %s" % (
                                                               title,
                                                               version,
                                                               updated_date,
                                                               )
                file_path = ''
                try:
                    file_path = tank.find(sinfo_address).system.filesystem_location
                except tank.common.errors.TankNotFound:
                    continue
                if file_path.startswith('/drd/'):
                    tab_data.append((
                                     display_title, file_path
                                     ))
            if len(tab_data) == 0:
                continue
            collection.append((tab_name, tab_data))
        amount = sum([len(entry[1]) for entry in collection])
        _time_taken = "%d min %d sec" % (amount * 17 // 60 , amount * 40 % 60)
        answer = QtGui.QMessageBox.question(self, 'Breakdown version comparison',
                                            'You are going to generate the breakdown-playlist version comparison '\
                                            'document.\nIt will be put in a folder on your desktop:\n'\
                                            '~/Desktop/breakdown_diagnosis\n\n'\
                                            '    Total shots to be processed: %d.\n    Time estimation: %s.\n\nDo you wish to continue?' % \
                                            (amount, _time_taken),
                                            QtGui.QMessageBox.Yes,
                                            QtGui.QMessageBox.No)
        if answer != QtGui.QMessageBox.Yes:
            return None
        print "\n-----------------------------\n" \
                "running breakdown diagnosis!" \
              "\n-----------------------------\n"
        result = bkdDiagn.sg_bkdDiagn.batch_process2(collection)
        html_file_name = time.ctime().replace(' ', '_').replace(':', '-') + ".html"
        html_file_dir = "/drd/users/%s/Desktop/breakdown_diagnosis" % getpass.getuser()
        if not os.path.exists(html_file_dir):
            os.mkdir(html_file_dir)
        html_file_path = html_file_dir + "/" + html_file_name
        open(html_file_path, 'w').write(result)
        QtGui.QMessageBox.information(self, 'Done',
                                      'Document has been generated:\n'\
                                      '%s' % html_file_path,
                                      QtGui.QMessageBox.Ok)

    '''
    ============================================

    clean up

    ============================================
    '''
    #    clean up the unused resources
    def handle_destroy(self):
#        print "cleaning the temp root %s" % THUMB_TEMP_ROOT
#        import shutil
#        for item  in os.listdir(THUMB_TEMP_ROOT):
#            fullpath = os.path.join(THUMB_TEMP_ROOT, item)
#            if os.path.isdir(fullpath):
#                shutil.rmtree( fullpath )
#            else:
#                os.remove(fullpath)
#        print "closing rv session"

#        if self._rv_session!=[]:
#            for rvs in self._rv_session:
#                if rvs.is_connected():
#                    print '.......disconnecting'
#                    rvs._rvs.disconnect()
#                    rvs.eval("close()")

        rv_tools.close_all_rv_binding()

        print "Saving preferences for review tool."
        self.pref_saver.save()

    def closeEvent(self, event):
        self.handle_destroy()



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

