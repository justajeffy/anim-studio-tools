##
#   \namespace  reviewTool.resources
#
#   \remarks    Rebuilding of the Review Tool system for reviewing shots and playlists from shotgun
#   
#   \author     Dr. D Studios
#   \date       07/26/11
#

import os.path

ICON_CACHE      = {}
PIXMAP_CACHE    = {}

from PyQt4.QtGui import QIcon, QPixmap

def find(relpath):
    """
    Looks up a resource given the current filepath and the inputed
    relative path location
    
    :param  relpath     <str>
    
    :return <str>
    """
    return os.path.join(os.path.dirname(__file__),relpath)
    
def icon(relpath):
    """
    Creates an icon and returns it as part of a singular cache to be reused
    
    :param relpath:     <str>
    
    :return <QIcon>
    """
    icon = ICON_CACHE.get(relpath)
    if ( not icon ):
        icon = QIcon(find(relpath))
        ICON_CACHE[relpath] = icon
    return icon

def pixmap(relpath):
    """
    Creates an pixmap and returns it as part of a singular cache to be reused
    
    :param relpath:     <str>
    
    :return <QPixmap>
    """
    pixmap = PIXMAP_CACHE.get(relpath)
    if ( not pixmap ):
        pixmap = QPixmap(find(relpath))
        PIXMAP_CACHE[relpath] = pixmap
    return pixmap

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

