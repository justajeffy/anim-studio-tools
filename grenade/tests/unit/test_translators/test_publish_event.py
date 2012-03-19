#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.publish_event import PublishEventTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestPublishEventTranslator(object):
    """
    Nose unit test suite for Grenade PublishEventTranslator.
    
    .. versionadded:: v00_04_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_04_00
        """
        self.shotgun_data = [{'id':1, 'type':'Project', 'sg_short_name':'hf2'},
                             {'id':2, 'type':'Scene', 'code':'19d', 'project':{'id':1, 'type':'Project'}},
                             {'id':3, 'type':'Shot', 'code':'19d_010', 'project':{'id':1, 'type':'Project'}},
                             {'id':4, 'type':'Asset', 'code':'mumble'},
                             {'id':5, 'type':'Asset', 'code':'mumbleDry'},
                             {'id':6, 'type':'Asset', 'code':'crevasse'},
                             {'id':7, 'type':'Asset', 'code':'empLand'},
                             {'id':8, 'type':'Asset', 'code':'bubbleTrail'},
                             {'id':9, 'type':'Asset', 'code':'bubbles'},
                             {'id':10, 'type':'Asset', 'code':'shiny_rock'},
                             {'id':11, 'type':'Asset', 'code':'clouds'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = PublishEventTranslator(self.session)
    
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
        verify_translate(self.translator, 'project', {'id':1, 'type':'Project'}, 'hf2', 'mm4')
        verify_translate(self.translator, 'sg_scene', {'id':2, 'type':'Scene'}, 'hf2:19d', 'mm4:19d')
        verify_translate(self.translator, 'sg_shot', {'id':3, 'type':'Shot'}, 'hf2:19d_010', 'mm4:19d_010')
        verify_translate(self.translator, 'sg_character', {'id':4, 'type':'Asset'}, 'mumble', 'gloria')
        verify_translate(self.translator, 'sg_environment', {'id':7, 'type':'Asset'}, 'empLand', 'adeleLand')
        verify_translate(self.translator, 'sg_fx', {'id':9, 'type':'Asset'}, 'bubbles', 'sparkles')
        verify_translate(self.translator, 'sg_fx_parent', {'id':8, 'type':'Asset'}, 'bubbleTrail', 'sparkleStick')
        verify_translate(self.translator, 'sg_prop', {'id':10, 'type':'Asset'}, 'shiny_rock', 'boring_rock')
        verify_translate(self.translator, 'sg_skydome', {'id':11, 'type':'Asset'}, 'clouds', 'blue_sky')
        verify_translate(self.translator, 'sg_stage', {'id':6, 'type':'Asset'}, 'crevasse', 'doomberg')
        verify_translate(self.translator, 'sg_surf_var', {'id':5, 'type':'Asset'}, 'mumbleDry', 'mumbleWetUnderwater')

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

