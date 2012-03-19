#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.task import TaskTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestTaskTranslator(object):
    """
    Nose unit test suite for Grenade TaskTranslator.
    
    .. versionadded:: v00_04_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v04_02_00
        .. versionchanged:: 1.3.0
            Updated pipeline step test data to include entity type.
        """
        self.shotgun_data = [{'id':1, 'type':'Group', 'code':'vfx'},
                             {'id':2, 'type':'HumanUser', 'login':'luke.cole'},
                             {'id':3, 'type':'Project', 'sg_short_name':'hf2'},
                             {'id':4, 'type':'Note'},
                             {'id':5, 'type':'Step', 'code':'Anim', 'entity_type':'Task'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = TaskTranslator(self.session)
    
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
        .. versionchanged:: 1.3.0
            Updaed pipeline step definition to include entity type.
        """
        verify_translate(self.translator, 'project', {'id':3, 'type':'Project'}, 'hf2', 'mm4')
        verify_translate(self.translator, 'sg_lead', {'id':2, 'type':'HumanUser'}, 'luke.cole', 'mark.streatfield')
        verify_translate(self.translator, 'sg_dept_lead', {'id':2, 'type':'HumanUser'}, 'luke.cole', 'mark.streatfield')
        verify_translate(self.translator, 'notes', [{'id':4, 'type':'Note'}], [{'Note':[['id', 'is', 4]]}], [{'Note':[['id', 'is', 5]]}])
        verify_translate(self.translator, 'open_notes', [{'id':4, 'type':'Note'}], [{'Note':[['id', 'is', 4]]}], [{'Note':[['id', 'is', 5]]}])
        verify_translate(self.translator, 'step', {'id':5, 'type':'Step'}, 'Task:Anim', 'Comp')
        verify_translate(self.translator, 'sg_group', {'id':1, 'type':'Group'}, 'vfx', 'comp')  
        verify_translate(self.translator, 'task_assignees', [{'id':1, 'type':'Group'}, {'id':2, 'type':'HumanUser'}], ['luke.cole', 'vfx'], ['anim'])
        verify_translate(self.translator, 'addressings_cc', [{'id':1, 'type':'Group'}, {'id':2, 'type':'HumanUser'}], ['luke.cole', 'vfx'], ['anim'])

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

