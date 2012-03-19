#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from .entity import EntityTranslator

from ..converters.default import convert_asset, convert_project, convert_scene, convert_shot

class PublishEventTranslator(EntityTranslator):
    """
    PublishEvent property translator.
    
    Assigning this translator to a PublishEvent model will cause inbound property values to be converted.
    
    .. versionadded:: v00_04_00
    """
      
    def __init__(self, session=None):
        """
        Setup (register converters, etc) the new translator instance.

        :param session:
            An active Shotgun session.
        
        .. versionadded:: v00_04_00
        
        .. todo::
            There are a lot of additional fields on this entity that could be translated if we had support
            for an asset converter.
        """
        EntityTranslator.__init__(self, session)

        self.register('project', convert_project)
        self.register('sg_scene', convert_scene)
        self.register('sg_shot', convert_shot)
        self.register('sg_character', convert_asset)
        self.register('sg_environment', convert_asset)
        self.register('sg_fx', convert_asset)
        self.register('sg_fx_parent', convert_asset)
        self.register('sg_prop', convert_asset)
        self.register('sg_skydome', convert_asset)
        self.register('sg_stage', convert_asset)
        self.register('sg_surf_var', convert_asset)

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

