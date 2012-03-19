#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from .entity import EntityTranslator

from ..converters.default import convert_project, convert_user, convert_recipients, convert_links

class NoteTranslator(EntityTranslator):
    """
    Note property translator.
    
    Assigning this translator to a Note model will cause inbound property values to be converted.
    
    .. versionadded:: v00_02_00
    .. versionchanged:: v00_03_00
        Removed obsolete internal conversion methods, update to support grenade.common.translator.Translator
        changes.
    .. versionchanged:: v00_04_00
        Inherit from the grenade.translators.entity.EntityTranslator to gain access to standard Shotgun entity translation behaviour
    """
      
    def __init__(self, session=None):
        """
        Setup (register converters, etc) the new translator instance.

        :param session:
            An active Shotgun session.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_03_00
            Use the default converters where relevant.
        .. versionchanged:: v00_04_00
            Updated to utilise EntityTranslator.
        """
        EntityTranslator.__init__(self, session)

        self.register('project', convert_project)
        self.register('user', convert_user)
        self.register('addressings_to', convert_recipients)
        self.register('addressings_cc', convert_recipients)
        self.register('note_links', convert_links)
        self.register('custom_entity10_sg_notes_custom_entity10s', convert_links)

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

