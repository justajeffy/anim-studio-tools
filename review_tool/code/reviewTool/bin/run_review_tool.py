import      sys, os, shutil, re, time, traceback
from        PyQt4                           import QtGui, QtCore
import      drGadgets
from        drGadgets.gui.DrSplash import DrSplash
import      reviewTool.gui.resource

TEST = False


def main(argv):
    # preference migration, this is to be remove after a while,
    # with new release, the preferences are stored per user per machine
    # migrate old preference
    old_pref_file = "/tmp/dailies_tool.conf"
    pref_file = "/tmp/review_tool.%s.conf" % os.environ.get("USER","general")
    if not os.path.isfile(pref_file) and os.path.isfile(old_pref_file):
        shutil.copyfile(old_pref_file, pref_file)

    app = QtGui.QApplication(sys.argv)
    splash = DrSplash(title="Review Tool",
                      version="Ver "+ os.environ.get("REVIEW_TOOL_VER", ""),
                      icon_path = reviewTool.gui.resource.getImage("applications-multimedia.png")
                      )

    splash.log("loading Tank...")
    import tank

    from reviewTool import TANK_VFS_DISABLE

    if not(TANK_VFS_DISABLE):
        splash.log("initialize Tank virtual file system...")
        tank.local.Tm()


    splash.log("initializing shotgun...")
    from drTank.util import shotgun_session
    shotgun_session()

    splash.log("initializing gui...")
    from reviewTool.gui.ContextWidget   import ContextWidget
    cw = ContextWidget(None)

    cw.show()
    splash.finish(cw)

    cw.splitter_2.setSizes([480,320])

    QtGui.QApplication.processEvents()

    cw.logger.info("Review tool startup success.")

    if len(argv)>1 and argv[1].startswith('playlist='):

        id_list = argv[1].replace('playlist=', '').split(',')

        for pid in id_list:
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                cw.populate_mix_master_from_shotgun( playlist_id=int(pid) )
            except:
                print "failed to load playlist with id", pid

            QtGui.QApplication.restoreOverrideCursor()

    elif len(argv)>1 and argv[1].startswith('cliplist='):

        id_list = argv[1].replace('cliplist=','').split(',')
        id_list = [ int(id) for id in id_list ]

        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            cw.populate_mix_master_from_shotgun( clip_id_list=id_list )
        except:
            traceback.print_exc()
            print "failed to load clips with ids", id_list

        QtGui.QApplication.restoreOverrideCursor()

    import platform
    if platform.system()=="Linux":
        rvexe = os.environ.get("RV_EXE")
        os.environ["RV_EXE"] = "%s -init /drd/software/ext/rv/lin64/config/drd_init " % rvexe

        #os.environ["RV_EXE"] = "/drd/software/ext/rv/lin64/rv-Linux-x86-64-3.10.10/bin/rv"
        # /drd/software/ext/rv/lin64/config/drd_init_3.8.6_anim.mu"


    '''-----------
      test block
    -----------'''
    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    main(sys.argv)


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

