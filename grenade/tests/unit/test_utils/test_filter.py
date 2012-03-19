#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_raises, assert_true

from grenade.utils import filter

class TestFilter(object):
    """
    Nose unit test suite for Grenade Shotgun Filter.

    .. versionadded:: 1.6.0
    """

    def test_convert(self):
        """
        Test that the convert function is correctly working.

        .. versionadded:: 1.6.0
        """

        filter1 = [['sg_status_1', 'is', ['apr', 'nfr']]]
        filter2 = [[['sg_status_1', 'is', 'apr'], ['sg_status_1', 'is', 'nfr']]]
        filter3 = [[['sg_status_1', 'is', 'apr'], ['sg_status_1', 'is', 'nfr'], filter.OPERATOR_AND]]

        result1 = result2 = {'conditions': [{'conditions': [{'values': ['apr'], 'path': 'sg_status_1', 'relation': 'is'}, {'values': ['nfr'], 'path': 'sg_status_1', 'relation': 'is'}], 'logical_operator': 'or'}], 'logical_operator': 'and'}
        result3 = {'conditions': [{'conditions': [{'values': ['apr'], 'path': 'sg_status_1', 'relation': 'is'}, {'values': ['nfr'], 'path': 'sg_status_1', 'relation': 'is'}], 'logical_operator': 'and'}], 'logical_operator': 'and'}

        assert_equals(filter.convert(filter1), result1)
        assert_equals(filter.convert(filter2), result2)
        assert_equals(filter.convert(filter3), result3)

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

