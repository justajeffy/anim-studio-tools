#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.reply import ReplyTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestReplyTranslator(object):
    """
    Nose unit test suite for Grenade ReplyTranslator.
    
    .. versionadded:: v00_06_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_06_00
        """
        self.shotgun_data = [{'id':3, 'type':'Note'}]        
        
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = ReplyTranslator(self.session)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_06_00
        """
        pass
    
    def test_translate(self):
        """
        Test that the translator converts the supplied test data as expected.
        
        .. versionadded:: v00_06_00
        """
        verify_translate(self.translator, 'entity', {'id':3, 'type':'Note'}, {'Note':[['id', 'is', 3]]}, {'Note':[['id', 'is', 4]]})

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

