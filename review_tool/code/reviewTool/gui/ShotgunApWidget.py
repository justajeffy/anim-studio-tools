import  sys,os

from    PyQt4           import      QtCore, QtGui, uic

import  tank
from    drTank.util     import      shotgun_session as ss
import  resource
import          drGadgets.lib.qtGuiPreference   as guiPrefSaver


shotgunAp   = uic.loadUiType(os.path.join(os.path.split(__file__)[0], "resource", "shotgunApWidget.ui"))[0]
department_filter_widget = uic.loadUiType(os.path.join(os.path.split(__file__)[0], "resource", "departmentFilterWidget.ui"))[0]

class ShotgunApWidget(shotgunAp, QtGui.QDialog):
    def __init__(self, pref_saver):
        context_default_dept = 0

        '''------------------------
                 setup UI
        ------------------------'''
        QtGui.QDialog.__init__(self)
        self.setupUi(self)
        self.tableWidget.setColumnWidth(0, 40)
        self.tableWidget.setColumnWidth(1, 90)
        self.tableWidget.setColumnWidth(2, 220)
        self.tableWidget.setColumnWidth(3, 180)
        self.tableWidget.setColumnWidth(4, 260)

        self.setWindowIcon ( resource.getIcon("shotgun_c.png"))
        self.tableWidget.verticalHeader().hide()
        self.resize(QtCore.QSize(1000,600))

        '''------------------------
             department filter
        ------------------------'''
        from ContextWidget import ALL_DEPARTMENTS, DEPARMENT_LABELS
        import copy
        ALL_DEPARTMENTS = copy.copy(ALL_DEPARTMENTS)
        DEPARMENT_LABELS = copy.copy(DEPARMENT_LABELS)

        ALL_DEPARTMENTS.sort()
        ALL_DEPARTMENTS.insert(0, "alldepts")
        DEPARMENT_LABELS["alldepts"] = "All Department"

        self.comboBox_dept.clear()
        self._shotgun_id = None

        i = 0
        for i in range(len(ALL_DEPARTMENTS)):
            dept = ALL_DEPARTMENTS[i]
            self.comboBox_dept.addItem( DEPARMENT_LABELS[dept] if DEPARMENT_LABELS.has_key(dept) else dept)
            self.comboBox_dept.setItemData(i,
                                           QtCore.QVariant(dept),
                                           QtCore.Qt.ToolTipRole)


        '''------------------------
             start-up behavior
        ------------------------'''
        self.refill_from_shotgun()

        self.pref_saver = pref_saver

        self.pref_saver.bootstrapWidget(      widget      = self.comboBox_dept,
                                         widgetName  = "comboBox_dept",
                                         widgetType  = guiPrefSaver.COMBOBOX )

        self._display_department_rows()


        self.tableWidget.setSelectionMode( 1 )
        self.tableWidget.setCurrentCell(0,0)
        self.header = self.tableWidget.horizontalHeader()

        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.sortItems(3, 1)

        '''------------------------
               connections
        ------------------------'''
        self.connect(
                     self.comboBox_dept,
                     QtCore.SIGNAL("currentIndexChanged ( int )"),
                     self._display_department_rows
                     )


        self.connect(
                     self.tableWidget,
                     QtCore.SIGNAL("itemSelectionChanged()"),
                     self.pick_up_id
                     )

        self.connect(
                     self.cancel_button,
                     QtCore.SIGNAL("clicked()"),
                     self,
                     QtCore.SLOT("reject()")
                     )

        self.connect(
                     self.ok_button,
                     QtCore.SIGNAL("clicked()"),
                     self,
                     QtCore.SLOT("accept()")
                     )

        self.connect(
                    self.tableWidget,
                    QtCore.SIGNAL("cellDoubleClicked (int,int)"),
                     self,
                     QtCore.SLOT("accept()")
                    )

#        self.connect(self, QtCore.SIGNAL("destroyed ()"), self.handle_destroy)

    def done(self, event=None):
        """
        Override the QWidget close function to save preferences upon exit.
        """
        self.pref_saver.save()
        QtGui.QDialog.done(self, event)


    def closeEvent(self, event):
        self.pref_saver.save()
        QtGui.QDialog.closeEvent(self, event)


    # searches shotgun database, then lists all the available playlists
    def refill_from_shotgun(self):
        bg_color = QtGui.QColor(0,0,0)
        bg_color.setHsv(89, 24, 230)
        id_font = QtGui.QFont("Sans Serif", 9, 10)
        id_font.setBold(True)
        self.tableWidget.hideColumn(0)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)

        header_list = ["id", "sg_department", "code", "sg_date_and_time", "description"]
        api = ss().api()
        all_playlist = api.find(
                                "Playlist",
                                [],
                                header_list
                                )
        row_index = 0
        for each_playlist in all_playlist:
            self.tableWidget.insertRow(row_index)
            self.tableWidget.verticalHeader().resizeSection(row_index, 22)

            for column_index in range(5):
                wItem = QtGui.QTableWidgetItem()
                #wItem.setSizeHint(QtCore.QSize(-1, 20))

                # restrict the users from accidentally editing the value in the cell
                wItem.setFlags(  QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable  )

                self.tableWidget.setItem(row_index, column_index, wItem)

                # set text for each row, write "-" if the corresponding value in shotgun field says None or ''
                # be aware about [Department], [sg_date_and_time], they'r composed by 2 levels of dictionary
                try:
                    if column_index == 0:
                        id_string = str(each_playlist[header_list[column_index]])
                        if len(id_string) == 1:
                            id_string = "0" + id_string
                        self.tableWidget.item(row_index, column_index).setText( id_string )
                        self.tableWidget.item(row_index, column_index).setBackgroundColor( bg_color )
                        self.tableWidget.item(row_index, column_index).setFont(id_font)

                    elif  column_index == 1:
                        self.tableWidget.item(row_index, column_index).setText(
                                                                               each_playlist[header_list[column_index]]["name"]
                                                                               )
                    elif column_index == 3:
                        timeObject = each_playlist[  header_list[column_index]  ]
                        time_string = str(timeObject.date()) + "   " + str(timeObject.time())
                        self.tableWidget.item(row_index, column_index).setText( time_string )
                    else:
                        self.tableWidget.item(row_index, column_index).setText(
                                                                               each_playlist[ header_list[column_index]]
                                                                               )
                except:
                    self.tableWidget.item(row_index, column_index).setText( "............" )

            row_index += 1


    def pick_up_id(self):
        self._shotgun_id = int(
                           self.tableWidget.item(self.tableWidget.currentRow() , 0).text()
                           )

    # hide the rows excluded from the department indexes
    def _display_department_rows(self, index=None):
        if index==None:
            index = self.comboBox_dept.currentIndex()

        department_token = str(self.comboBox_dept.itemData(index, QtCore.Qt.ToolTipRole).toString())

        length = self.tableWidget.rowCount()

        if department_token == "alldepts":
            for i in range(length):
                self.tableWidget.showRow(i)
        else:

            for i in range(length):
                if str( self.tableWidget.item(i, 1).text() ) == department_token:
                    self.tableWidget.showRow(i)
                else:
                    self.tableWidget.hideRow(i)


class DepartmentFilterWidget(department_filter_widget , QtGui.QDialog):
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.connect(
                     self.button_up,
                     QtCore.SIGNAL("clicked()"),
                     self._move_row_up
                     )

        self.connect(
                     self.button_down,
                     QtCore.SIGNAL("clicked()"),
                     self._move_row_down
                     )
        self.setWindowTitle("Department Play Priority Order and Filter")
        self.button_up.setIcon(QtGui.QIcon(resource.getImage("up.png")))
        self.button_down.setIcon(QtGui.QIcon(resource.getImage("down.png")))
        self.resize(QtCore.QSize(500,500))

    def _get_selection(self):
        '''
        single selection atm
        '''
        if len(self.list.selectedItems()) != 0:  return self.list.selectedItems()[0]
        else:  return None

    def _move_row_up(self):
        self._move_row('up')
        self.emit(QtCore.SIGNAL("StateChanged"), self)
    def _move_row_down(self):
        self._move_row('down')
        self.emit(QtCore.SIGNAL("StateChanged"), self)
    def _move_row(self, direction):
        '''
        direction = 'up'  / 'down'
        this one moves the row in the GUI also adjusts the order of the list entries
        '''
        if self._get_selection() != None:
            if direction == "up" and self.list.currentRow() > 0:
                index = self.list.currentRow()
                item = self.list.takeItem(self.list.currentRow())
                self.list.insertItem(index-1, item)
                self.list.setItemSelected(item, True)
                self.list.setCurrentItem(item)

            elif direction == "down" and self.list.currentRow() < self.list.count() - 1:
                index = self.list.currentRow()
                item = self.list.takeItem(self.list.currentRow())
                self.list.insertItem(index+1, item)
                self.list.setItemSelected(item, True)
                self.list.setCurrentItem(item)



if __name__=="__main__":
    ap = QtGui.QApplication([])

    s = ShotgunApWidget()

    s.exec_()

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

