#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_true

from grenade.common.translator import Translator

class TestTranslator(object):
    """
    Nose unit test suite for Grenade Translator.
    
    .. versionadded:: v00_02_00
    """
    def setup(self):
        """
        Set up the unit test suite. Defines a test converter as a local method which gets registered with the translator.
        
        .. versionadded:: v00_02_00
        """
        self.translator = Translator(session=None)
        
        def test_converter(session, value):
            return str(value)
        
        self.translator.register('test', test_converter)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_02_00
        """
        pass
    
    def test_translate(self):
        """
        Test that the translate method works as expected (i.e., specified data is translated)
        
        .. versionadded:: v00_02_00
        """
        assert_equals(self.translator.session, None)
        assert_true(self.translator.converters.has_key('test'))
        assert_equals(len(self.translator.converters.keys()), 1)
        
        result = self.translator.translate('test', 123)
        
        assert_equals(result, '123')
        
    def test_register(self):
        """
        Test that the register method works as expected (i.e., registers the specified converter)
        
        .. versionadded:: v00_03_00
        """
        def test_converter(session, value):
            pass
        
        self.translator.register('convert', test_converter)
        
        assert_true(self.translator.converters.has_key('convert'))
        assert_equals(len(self.translator.converters.keys()), 2)
        assert_equals(self.translator.converters['convert'], test_converter)

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

