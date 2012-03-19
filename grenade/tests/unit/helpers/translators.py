#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_raises, assert_true

from grenade.common.error import GrenadeValidationError

def verify_translate(translator, field, result, good_value, bad_value):
    """
    Helper to simplify testing translation methods.
        
    .. versionadded:: v00_04_00
    .. versionchanged:: 0.11.0
        Update to use nose asserts statements.
    """
    output = translator.translate(field, good_value)
    
    if type(result) == list and len(result) > 1:
        assert_equals(len(output), len(result))
        for entity in result:
            assert_true(entity in result)
    else:
        assert_equals(output, result)
    
    assert_raises(GrenadeValidationError, translator.translate, field, bad_value)

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

