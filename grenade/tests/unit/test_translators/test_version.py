#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.version import VersionTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestVersionTranslator(object):
    """
    Nose unit test suite for Grenade ShotTranslator.
    
    .. versionadded:: 1.4.0
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 1.4.0
        """
        self.shotgun_data = [{'id':1, 'type':'Project', 'sg_short_name':'hf2'},
                             {'id':2, 'type':'HumanUser', 'login':'luke.cole'},
                             {'id':3, 'type':'Asset'},
                             {'id':4, 'type':'Note'},
                             {'id':5, 'type':'Shot'},
                             {'id':6, 'type':'Task'},
                             {'id':7, 'type':'PublishEvent'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = VersionTranslator(self.session)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 1.4.0
        """
        pass
    
    def test_translate(self):
        """
        Test that the translator converts the supplied test data as expected.
        
        .. versionadded:: 1.4.0
        """
        verify_translate(self.translator, 'entity', {'id':5, 'type':'Shot'}, {'Shot':[['id', 'is', 5]]}, {'Shot':[['id', 'is', 6]]})
        verify_translate(self.translator, 'notes', [{'id':4, 'type':'Note'}], [{'Note':[['id', 'is', 4]]}], [{'Note':[['id', 'is', 5]]}])
        verify_translate(self.translator, 'open_notes', [{'id':4, 'type':'Note'}], [{'Note':[['id', 'is', 4]]}], [{'Note':[['id', 'is', 5]]}])
        verify_translate(self.translator, 'project', {'id':1, 'type':'Project'}, 'hf2', 'mm4')
        verify_translate(self.translator, 'sg_assets', [{'id':3, 'type':'Asset'}], [{'Asset':[['id', 'is', 3]]}], [{'Asset':[['id', 'is', 4]]}])
        verify_translate(self.translator, 'sg_assigned_to', {'id':2, 'type':'HumanUser'}, 'luke.cole', 'anim')
        verify_translate(self.translator, 'sg_tank_revision', {'id':7, 'type':'PublishEvent'}, {'PublishEvent':[['id', 'is', 7]]}, {'PublishEvent':[['id', 'is', 8]]})
        verify_translate(self.translator, 'sg_tasks', [{'id':6, 'type':'Task'}], [{'Task':[['id', 'is', 6]]}], [{'Task':[['id', 'is', 7]]}])
        verify_translate(self.translator, 'user', {'id':2, 'type':'HumanUser'}, 'luke.cole', 'anim')

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

