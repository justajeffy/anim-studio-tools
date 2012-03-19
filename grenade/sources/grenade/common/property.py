#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from copy import deepcopy

from ..common.error import GrenadeValidationError

class Property(object):
    """
    Generic grenade model property.
    
    .. versionadded:: v00_01_00
    .. versionchanged:: v00_02_00
        Explicitly inherit from object.
    """
    
    def __init__(self, value, data_type=None, valid_values=[], read_only=False):
        """
        Setup the new property instance.

        :param value:
            The value that this property represents.        
  
        .. versionadded:: v00_01_00
        .. versionchanged:: v00_03_00
            Added two new fields to store the Shotgun data type of the property, and an optional list of valid values 
            (should also come from Shotgun).
        .. versionchanged:: v00_07_00
            Take a copy of the value to check against if it is of type multi_entity, because users can modify the
            contents without the property being aware.
        .. versionchanged:: v00_07_00
            Implement accessors for the is_dirty flag, take into account the check value, if present.
        .. versionchanged:: 0.9.0
            Added a new field to indicate whether the property is read-only or not.   This value is set by the 'editable'
            value of the associated Shotgun field, as recorded within the schema.
            
        .. warning::
            Even though we're detecting changes (eg., .append()) to multi_entity properties now via the check value, 
            we're still relying on the user to pass in the correct value - if a translator is attached to the model, 
            it won't be able to translate the value if the user doesn't set the field via the set_value() method ...
        """

        self.value = value    
        self.data_type = data_type
        self.valid_values = valid_values
        self.is_dirty = False
        self.is_read_only = read_only
        
        # it is possible for the user to edit the contents of this property, without us being aware that they have done so
        #     e.g., property.value.append(foo) doesn't update is_dirty flag (!)
        #
        # the only way for us to detect this is to take a copy of the value for us to check against later, to see if it has
        # been modified, in cases where this situation may occur (ie, for multi_entity values)
        
        self.check = deepcopy(value) if self.data_type == 'multi_entity' else None
    
    def reset_check(self):
        """
        Reset the check value.
        
        .. versionadded:: v00_07_00
        """
        if self.data_type == 'multi_entity':
            self.check = deepcopy(self.value)
    
    def get_is_dirty(self):
        """
        Get the current value of the is_dirty flag for this property.
        
        :returns:
            True or false, indicating whether the property has been changed.
            
        .. versionadded:: v00_07_00
        
        .. note::
            This method also takes into account the check value, if present.   It is possible for this method to return True,
            even if the is_dirty flag is False; this happens if the current value differs from the check value.
        """
        if self.is_dirty or (self.data_type == 'multi_entity' and self.check != self.value):
            return True
        
        return False
    
    def set_is_dirty(self, dirty):
        """
        Manually set the is_dirty flag for this property.   The is_dirty flag indicates whether the property has been updated.
        
        :param dirty:
            The (boolean) value to assign to the is_dirty flag.
            
        .. versionadded:: v00_07_00
        """
        self.is_dirty = dirty
        
    def set_value(self, value):
        """
        Set the value of the property.   Checks to see that the input is within the range of valid values (if set).
        Marks the property as dirty/modified.

        :param value:
            The value to assign to this property.
        
        .. versionadded:: v00_01_00
        .. versionchanged:: v00_03_00
            Raise an exception if the specified value is not valid, but only if a list of valid values has been set
            on this property.
        .. versionchanged:: v00_05_03
            None is a permitted 'list' data type value.
        .. versionchanged:: v00_05_04
            Explicitly match against None (False might be a valid_value for example ...)
            
        .. todo::
            Support checking of values against other, non-list data types.
        """
        if self.data_type == 'list' and self.valid_values and value != None and value not in self.valid_values:
            raise GrenadeValidationError('Invalid property value : %s not in %s' % (value, self.valid_values))
        
        self.value = value
        self.is_dirty = True
        
    def __repr__(self):
        """
        Output a generic representation of the property instance.

        :returns:
            Printable representation of self.
        
        .. versionadded:: v00_01_00
        """
        if isinstance(self.value, str):
            return "'%s'" % self.value
        else:
            return '%s' % self.value

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

