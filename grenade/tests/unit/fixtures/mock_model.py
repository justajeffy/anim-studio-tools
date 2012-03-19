#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from grenade.common.model import Model

class MockModel(Model):
    """
    Mock Grenade model for use within Grenade unit tests.
    
    .. versionadded:: v00_02_00
    """
    __schema__ = ['field_one', 'field_two', 'field_three']
    
    def __init__(self, session, identifier=None, translator=None, schema={'id':None}, **kwargs):
        """
        Setup the new mock model instance.

        :param session:
            An active Shotgun session.
        :param identifer:
            An identifier for this model.
        :param translator:
            Optional property translator (applies conversions to inbound property values).
        :param schema:
            Schema definition for this model.
        :param field_one:
            Test property.
        :param field_two:
            Test property.
        :param field_three:
            Test property.
        
        .. versionadded:: v00_02_00
        """
        Model.__init__(self, session, identifier, translator, schema)
        
        if kwargs:
            for key in kwargs.keys():
                self.set_property(key, kwargs[key])
        
        self._purify()  # make sure all the properties we've just set are marked as clean

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

