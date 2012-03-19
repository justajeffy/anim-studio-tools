#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.project import ProjectTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestProjectTranslator(object):
    """
    Nose unit test suite for Grenade ProjectTranslator.
    
    .. versionadded:: v00_04_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_04_00
        """
        self.shotgun_data = [{'id':1, 'type':'HumanUser', 'login':'luke.cole'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = ProjectTranslator(self.session)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_04_00
        """
        pass
    
    def test_translate(self):
        """
        Test that the translator converts the supplied test data as expected.
        
        .. versionadded:: v00_04_00
        """
        verify_translate(self.translator, 'users', [{'id':1, 'type':'HumanUser'}], [{'HumanUser':[['id', 'is', 1]]}], [{'HumanUser':[['id', 'is', 2]]}])

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

