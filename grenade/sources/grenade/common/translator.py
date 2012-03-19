#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

class Translator(object):
    """
    Superclass for a generic data translator.
    
    .. versionadded:: v00_02_00
    """
    
    def __init__(self, session=None):
        """
        Setup the new translator instance.

        :param session:
            An active Shotgun session.
        
        .. versionadded:: v00_02_00
        """

        self.session = session
        self.converters = {}

    def translate(self, key, value):
        """
        Translate the given value using the registered converter for the supplied key, if present.

        :param key:
            Identifier of the registered converter to use (if present).
        :param value:
            The value to be translated.
        :returns:
            The translated value, or value, if no converter matching the supplied key was found.
        
        .. versionadded:: v00_02_00
        """
        if key in self.converters.keys():
            return self.converters[key](self.session, value)
        else:
            return value
    
    def register(self, key, converter):
        """
        Register a conversion method against the supplied key.

        :param key:
            Identifier under which to register the converter.
        :param converter:
            Callable that accepts one parameter, the value to be translated.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_03_00
            This method is now no longer private to translator sub-classes.
        """
        self.converters[key] = converter

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

