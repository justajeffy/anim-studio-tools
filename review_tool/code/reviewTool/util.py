import os, re
#from rv_tools.util import get_frame_sequence_source as get_frame_sequence_source_fast
#from rv_tools.util import is_movie_path as is_movie
from rv_tools.util import RG_FILE_SEQUENCE

TMP_DIR = "/tmp/rv_temp"
if not(os.path.isdir(TMP_DIR)):
    os.mkdir(TMP_DIR)


def is_valid_render(path, render_type, check_filesize=True):
    # criteria for valid movie, file exist, and file bigger than 10 bytes
    # criteria for valid frames dir, one file in directory, and file bigger than 10 bytes
    #     the frames dir should  not contain any folders
    #     this is verify to the point if the first item is a folder, then it returns true

    if render_type=="Movie" and os.path.isfile( path ):
        if check_filesize and os.path.getsize(path) < 10:
            return False

        return True

    elif render_type=="Frames" and os.path.isdir(path):

        dir_files = os.listdir(path)
        if dir_files:
            MAX_TEST = 10
            test_count = 0
            for file_name in dir_files:
                test_count+=1

                if test_count> MAX_TEST: break

                full_path = os.path.join(path, file_name)

                if os.path.isfile(full_path) and RG_FILE_SEQUENCE.search(full_path):
                    return True

    return False


def get_tmp_rv_file_name(shotgun_playlist_id):
    # creating the temp directory
    if not os.path.exists("/tmp/rv_annotate"):
        os.mkdir("/tmp/rv_annotate")

    return "/tmp/rv_annotate/review_tool_playlist%s.%s.rv" % ("_shotgun_id_%s" % shotgun_playlist_id if shotgun_playlist_id else "",
                                                               os.environ.get("USER", "spidey"))

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

