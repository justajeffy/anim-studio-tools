#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.courier import CourierTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestCourierTranslator(object):
    """
    Nose unit test suite for Grenade CourierTranslator.
    
    .. versionadded:: 1.5.0
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 1.5.0
        """
        self.shotgun_data = [{'id':1, 'type':'Playlist', 'code':'test_playlist'},
                             {'id':2, 'type':'Project', 'sg_short_name':'hf2'},
                             {'id':3, 'type':'Version'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = CourierTranslator(self.session)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 1.5.0
        """
        pass
    
    def test_translate(self):
        """
        Test that the translator converts the supplied test data as expected.
        
        .. versionadded:: 1.5.0
        """
        verify_translate(self.translator, 'project', {'id':2, 'type':'Project'}, 'hf2', 'mm4')
        verify_translate(self.translator, 'sg_playlist', {'id':1, 'type':'Playlist'}, {'Playlist':[['id', 'is', 1]]}, {'Playlist':[['id', 'is', 2]]})
        verify_translate(self.translator, 'sg_image_sequences', [{'id':3, 'type':'Version'}], [{'Version':[['id', 'is', 3]]}], [{'Version':[['id', 'is', 4]]}])

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

