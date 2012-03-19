#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from .entity import EntityTranslator

from ..converters.default import convert_datetime, convert_recipients, convert_user, convert_links

class SpoolTranslator(EntityTranslator):
    """
    Spool property translator.
    
    Assigning this translator to a Spool model will cause inbound property values to be converted.
    
    .. versionadded:: 1.7.2
    """
      
    def __init__(self, session=None):
        """
        Setup (register converters, etc) the new translator instance.

        :param session:
            An active Shotgun session.
        
        .. versionadded:: 1.7.2
        
        .. todo::
            Replace convert_links usage with more entity specific converters (?)
        """
        EntityTranslator.__init__(self, session)

        self.register('sg_checksum_qa', convert_recipients)
        self.register('sg_delivery_time', convert_datetime)
        self.register('sg_edit_qa', convert_recipients)
        self.register('sg_delivered_by', convert_user)
        self.register('sg_tank_created_by', convert_user)
        self.register('sg_tank_creation_date', convert_datetime)
        self.register('sg_visual_qa', convert_recipients)
        self.register('sg_notes', convert_links)

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

