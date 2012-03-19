SUPPORTED_IMAGE = ("jpg", "png", "exr", "iff")
SUPPORTED_MOVIE = ("mov", "mp4")
import os, subprocess, traceback, time
import platform

def get_image_thumbnail_rvio(path, thumbWidth, thumbHeight, thumbTmpRoot):

    # generate the thumb using rv
    thumbName = os.path.split(path)[1] + ".jpg"
    thumbPath = os.path.join(thumbTmpRoot, thumbName)

    oxcmd = "/drd/software/ext/rv/osx/3.10.11/RV.app/Contents/MacOS/rvio " + path +  " -outres " + str(thumbWidth) + ' ' + str(thumbHeight) + " -o " + thumbPath

    if not os.path.isfile(thumbPath):
        if platform.system()=="Linux":
            cmd = "rvio " + path +  " -outres " + str(thumbWidth) + ' ' + str(thumbHeight) + " -o " + thumbPath
            ret = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            ret.communicate()
        else:
            cmd = "/drd/software/ext/rv/osx/3.10.11/RV.app/Contents/MacOS/rvio " + path +  " -outres " + str(thumbWidth) + ' ' + str(thumbHeight) + " -o " + thumbPath
            ret = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            start_ = time.time()
            while (time.time() - start_ < 1.5) and not os.path.isfile(thumbPath):
               time.sleep(0.05)


    if os.path.isfile(thumbPath):
        return thumbPath
    else:
        return None



def get_image_thumbnail(path, thumbWidth, thumbHeight, thumbTmpRoot="/tmp"):
    file_ext = path.split(".")[-1]

    img_path =  get_image_thumbnail_rvio(path, thumbWidth, thumbHeight, thumbTmpRoot)

    return img_path



if __name__=="__main__":
    img = read_iff_via_libImage("/tmp/test.iff", 90, 60)
    img.save("/tmp/iff_to_jpeg.jpg")


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

