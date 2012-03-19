import os, sys, time, traceback, subprocess

from PyQt4 import QtGui, QtCore, uic

from resource import getIcon

render_rev_panel    = uic.loadUiType(os.path.join(os.path.split(__file__)[0], "render_revision_row.ui"))[0]

INFO_ICON = getIcon("exclamation_navy.png")

class RevMenuItem(render_rev_panel, QtGui.QWidget):

    def __init__(self, parent, row_item=None, prefSaver=None, dialog=None):
        QtGui.QWidget.__init__(self, parent)
        self.row_item = row_item

        self.setupUi(self)
        self._meta_data = None
        self._shotgun_id = None
        self.tank_button.hide()
        self._dialog = dialog

        self.connect(self.shotgun_button, QtCore.SIGNAL("clicked()"), self.handle_show_in_shotgun)
        self.shotgun_button.setIcon(INFO_ICON)
        self.setMouseTracking(True)

    def enterEvent(self, event):

        self.setAutoFillBackground(True)

        p = self.palette()

        self.highlight_color = QtGui.QColor(255, 218, 51, 255)
        p.setBrush(QtGui.QPalette.Active,   QtGui.QPalette.Base, QtGui.QBrush(self.highlight_color))

        self.setPalette(p)
        QtGui.QWidget.enterEvent(self, event)


    def leaveEvent(self, event):

        self.setAutoFillBackground(False)
        p = self.palette()

        self.highlight_color = QtGui.QColor(0, 0, 0, 0) #155, 161, 158, 255)
        p.setBrush(QtGui.QPalette.Active,   QtGui.QPalette.Base, QtGui.QBrush(self.highlight_color))
        self.setPalette(p)

        QtGui.QWidget.leaveEvent(self, event)


    def handle_show_in_shotgun(self):
        cmd = 'exo-open http://shotgun/detail/Version/%s' % self._shotgun_id
        p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
        self._dialog.close()


class RevMenuDialog(QtGui.QDialog):
    def __init__(self, *argList):
        QtGui.QDialog.__init__(self, *argList)

        l = QtGui.QVBoxLayout()
        l.setMargin(0)
        l.setSpacing(0)
        self.tree = QtGui.QTreeWidget(self)
        
        self._user_modified = False

        l.addWidget(self.tree)
        #l.addWidget(QtGui.QLabel("Check/uncheck one or more clips, click outside to close."))

        self.setLayout(l)
        self.tree.header().hide()
        self.tree.setRootIsDecorated ( False)

        self.tree.setAlternatingRowColors(True)

        self.connect(   self.tree,
                        QtCore.SIGNAL("itemPressed ( QTreeWidgetItem *, int)"),
                        self._handle_selection_change )


#        palette = QtGui.QPalette()
#        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
#        brush.setStyle(QtCore.Qt.SolidPattern)
#        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
#        brush = QtGui.QBrush(QtGui.QColor(200, 200, 200))
#        brush.setStyle(QtCore.Qt.SolidPattern)
#        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
#        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
#        brush.setStyle(QtCore.Qt.SolidPattern)
#        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
#        brush = QtGui.QBrush(QtGui.QColor(200, 200, 200))
#        brush.setStyle(QtCore.Qt.SolidPattern)
#        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
#        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
#        brush.setStyle(QtCore.Qt.SolidPattern)
#        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
#        brush = QtGui.QBrush(QtGui.QColor(200, 200, 200))
#        brush.setStyle(QtCore.Qt.SolidPattern)
#        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
#        self.tree.setPalette(palette)

        self._state_handle_selection_change = False


    def _handle_selection_change(self, current, column):
        if self._state_handle_selection_change:
            return
        rowItem = self.tree.itemWidget(current, 0)
        
        if rowItem.name_checkbox.isChecked():
            rowItem.name_checkbox.setChecked(False)
        else:
            rowItem.name_checkbox.setChecked(True)

        self.close()
        self.set_user_modified()

        self._state_handle_selection_change = False
    
    def is_user_modified( self ):
        return self._user_modified

    def list_checked_items(self):
        item_list = []
        for i in range ( self.tree.topLevelItemCount() ):
            menu_item = self.tree.itemWidget(self.tree.topLevelItem(i), 0)
            if menu_item.name_checkbox.isChecked():
                item_list.append( menu_item )

        return item_list

    def add_item(self):
        row_item = QtGui.QTreeWidgetItem( self.tree.invisibleRootItem() )
        item = RevMenuItem( self.tree, row_item= row_item, dialog=self)
        self.connect(item.name_checkbox,QtCore.SIGNAL('clicked()'),self.set_user_modified)
        self.tree.setItemWidget(row_item, 0, item)

        return item
    
    def set_user_modified( self, state = True ):
        self._user_modified = state


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

