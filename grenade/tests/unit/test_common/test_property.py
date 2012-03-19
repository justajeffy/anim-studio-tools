#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_not_equals, assert_raises, assert_true

from grenade.common.error import GrenadeValidationError
from grenade.common.property import Property

class TestProperty(object):
    """
    Nose unit test suite for Grenade Property.
    
    .. versionadded:: v00_02_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_03_00
            Updated to set the property data_type and valid_values
        """
        self.property = Property('test_value', data_type='list', valid_values=['test_value', 'updated_value'])
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_02_00
        """
        pass
    
    def test_init(self):
        """
        Test that the property is initialised correctly.
        
        .. versionadded:: 0.9.0
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        assert_equals(self.property.value, 'test_value')
        assert_equals(self.property.data_type, 'list')
        assert_equals(self.property.valid_values, ['test_value', 'updated_value'])
        assert_equals(self.property.is_dirty, False)
        assert_equals(self.property.is_read_only, False)
        assert_equals(self.property.check, None)
    
    def test_get_is_dirty(self):
        """
        Test that the get_is_dirty() accessor behaves as expected.
        
        .. versionadded:: v00_07_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        assert_equals(self.property.get_is_dirty(), False)
        self.property.set_value('updated_value')
        assert_equals(self.property.get_is_dirty(), True)
        
    def test_set_is_dirty(self):
        """
        Test that the set_is_dirty() accessor behaves as expected.
        
        .. versionadded:: v00_07_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        assert_equals(self.property.get_is_dirty(), False)
        self.property.set_is_dirty(True)
        assert_equals(self.property.get_is_dirty(), True)
    
    def test_reset_check(self):
        """
        Test that the reset_check() method updates the check field as expected.
        
        .. versionadded:: v00_07_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        self.property = Property([1], data_type='multi_entity')
        assert_equals(self.property.check, self.property.value)
        self.property.value.append(2)
        assert_not_equals(self.property.check, self.property.value)
        self.property.reset_check()     # users should not need to do this under normal circumstances ...
        assert_equals(self.property.check, self.property.value)
    
    def test_set_value(self):
        """
        Test that the Property value and is_dirty fields are correctly updated when the set_value() method is called.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_03_00
            Added test code to verify an error is thrown if an invalid value is provided.
        .. versionchanged:: v00_05_03
            Added test code to verify that None is an allowable value for a list data type.
        .. versionchanged:: v00_07_00
            Use the new Property accessors for the is_dirty field.
        .. versionchanged:: v00_07_00
            Ensure that the Property correctly detects external updates to its value.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        assert_equals(self.property.get_is_dirty(), False)
        assert_equals(self.property.value, 'test_value')
        
        self.property.set_value('updated_value')
        
        assert_equals(self.property.get_is_dirty(), True)
        assert_equals(self.property.value, 'updated_value')
        
        # nose lamely has no assert_not_raises() ...
        try:
        	self.property.set_value(None)
        	assert_true(True)
        except GrenadeValidationError, e:
        	assert_true(False)
        
        assert_raises(GrenadeValidationError, self.property.set_value, 'invalid_value')
            
        self.property = Property([1], data_type='multi_entity')
        self.property.value.append(2)
        assert_equals(self.property.value, [1, 2])
        assert_equals(self.property.get_is_dirty(), True)
        
    def test_repr(self):
        """
        Test that the Property representation is generated correctly for strings and other types of value.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        # test string specific representation
        result = self.property.__repr__()
        
        assert_equals(result, "'test_value'")
        
        # test default representation
        self.property = Property(123)
        result = self.property.__repr__()
        
        assert_equals(result, '123')

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

