import pimath as p
import napalm.core as n
import napalmImageIO as ni
import math, os

def createImage():
    w = 320
    h = 240
    b = n.V3fBuffer(w * h, p.V3f(0, 0, 0))

    print '# creating pixel data'
    for y in range(h):
        for x in range(w):
            s = float(x) / w
            t = float(y) / h
            b[y * w + x] = p.V3f(s, t, 0)

    img = n.ObjectTable()
    img['xres'] = w
    img['yres'] = h
    img['pixels'] = b

    # check that this is a valid napalm image
    assert ni.isValid(img, True)

    return img

def diffPixelData(lista, listb):
    assert len(lista) == len(listb)
    for i in range(len(lista)):
        pa = lista[i]
        pb = listb[i]
        assert math.fabs(pa.x - pb.x) < 0.001
        assert math.fabs(pa.y - pb.y) < 0.001
        assert math.fabs(pa.z - pb.z) < 0.001

def verifyImage(a, b):
    assert ni.isValid(a, True)
    assert ni.isValid(b, True)
    assert a['xres'] == b['xres']
    assert a['yres'] == b['yres']
    diffPixelData(a['pixels'].contents, b['pixels'].contents)

file_path = '/tmp/test_grad.png'
ref_path = './test_grad.png'

print '# removing old test data'
if os.path.exists(file_path):
    os.remove(file_path)

a = createImage()

print '# writing image to %s' % file_path
ni.write(a, file_path, 'uchar')

print '# reading image from %s' % file_path
b = ni.read(file_path)

print '# verifying image data matches'
verifyImage(a, b)

print '# viewing result'
os.system('hview %s &' % (file_path))

print '# checking result'
os.system('perceptualdiff -colorfactor 1.0 -threshold 10 %s %s'%(file_path,ref_path))

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

