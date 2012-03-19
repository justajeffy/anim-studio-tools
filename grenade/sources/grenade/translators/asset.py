#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from .entity import EntityTranslator

from ..converters.default import convert_links, convert_project, convert_scene

class AssetTranslator(EntityTranslator):
    """
    Asset property translator.
    
    Assigning this translator to an Asset model will cause inbound property values to be converted.
    
    .. versionadded:: v00_04_00
    """
      
    def __init__(self, session=None):
        """
        Setup (register converters, etc) the new translator instance.

        :param session:
            An active Shotgun session.
        
        .. versionadded:: v00_04_00
        
        .. todo::
            Replace convert_links usage with more entity specific converters (?)
        """
        EntityTranslator.__init__(self, session)

        self.register('project', convert_project)
        self.register('sg_1st_scene_appearance', convert_scene)
        self.register('scenes', convert_links)
        self.register('sequences', convert_links)
        self.register('shots', convert_links)
        self.register('tasks', convert_links)
        self.register('assets', convert_links)                      # SubAssets

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

