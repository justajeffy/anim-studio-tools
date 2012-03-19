import sys, os
from PyQt4 import QtCore, QtGui

import tank
from drGadgets.lib import qtGuiPreference

from MixMasterGui import MixMasterGui

if not(os.path.isdir("/tmp/mixmaster")):
    os.mkdir("/tmp/mixmaster")

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    flgOpenMixMaster = True

    if flgOpenMixMaster:
        prefSaver = qtGuiPreference.QtGuiPreference("/tmp/mixMaster.conf")

        mmm = MixMasterGui(None, prefSaver, thumbTmpRoot = "/tmp/mixmaster")
        mmm.show()

        rvfile = [item for item in sys.argv if item.endswith(".rv")]

        if rvfile:
            mmm.importPlaylist(rvfile[0])

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

