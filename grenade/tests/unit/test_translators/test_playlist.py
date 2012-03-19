#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from datetime import datetime

from ..helpers.translators import verify_translate

from grenade.translators.playlist import PlaylistTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestPlaylistTranslator(object):
    """
    Nose unit test suite for Grenade PlaylistTranslator.
    
    .. versionadded:: v00_05_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_05_00
        """
        self.shotgun_data = [{'id':1, 'type':'Project', 'sg_short_name':'hf2'},
                             {'id':2, 'type':'CustomNonProjectEntity03', 'code':'anim'},
                             {'id':3, 'type':'Version', 'code':'sc_21a_010_lens_v009'},
                             {'id':4, 'type':'Version', 'code':'sc_21a_020_lens_v009'},
                             {'id':5, 'type':'CustomEntity01', 'code':'Important Meeting'},
                             {'id':6, 'type':'Note'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = PlaylistTranslator(self.session)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_05_00
        """
        pass
    
    def test_translate(self):
        """
        Test that the translator converts the supplied test data as expected.
        
        .. versionadded:: v00_05_00
        """
        verify_translate(self.translator, 'project', {'id':1, 'type':'Project'}, 'hf2', 'mm4')
        verify_translate(self.translator, 'versions', [{'id':3, 'type':'Version'}, {'id':4, 'type':'Version'}], ['sc_21a_010_lens_v009', 'sc_21a_020_lens_v009'], ['sc_21a_090_anim_v002'])
        verify_translate(self.translator, 'sg_department', {'id':2, 'type':'CustomNonProjectEntity03'}, 'anim', 'fx')
        verify_translate(self.translator, 'notes', [{'id':6, 'type':'Note'}], [{'Note':[['id', 'is', 6]]}], [{'Note':[['id', 'is', 7]]}])
        verify_translate(self.translator, 'meetings', [{'id':5, 'type':'CustomEntity01'}], ['Important Meeting'], ['Boring Meeting'])
        verify_translate(self.translator, 'sg_date_and_time', datetime(2010, 8, 30, 15, 0), '30/08/2010 15:00', '30-08-2010 15.00')

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

