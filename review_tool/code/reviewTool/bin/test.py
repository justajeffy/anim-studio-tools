from PyQt4 import QtGui, QtCore


class D(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.setWindowTitle("hello, world")


        t = QtCore.QTimer(self)
        self.connect(t, QtCore.SIGNAL("timeout()"), self.bring_to_top)
        t.start(1000)
        self.setWindowFlags( QtCore.Qt.WindowStaysOnTopHint )

        self.setWindowFlags( QtCore.Qt.WindowStaysOnTopHint )


    def bring_to_top(self):
        print "bring window to top"


if __name__ == "__main__":
    app = QtGui.QApplication([])
    d = D()

    d.exec_()

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

