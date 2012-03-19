#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.note import NoteTranslator
from probe.fixtures.mock_shotgun import MockShotgun

class TestNoteTranslator(object):
    """
    Nose unit test suite for Grenade NoteTranslator.
    
    .. versionadded:: v00_02_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_02_00
        """
        self.shotgun_data = [{'id':1, 'type':'Group', 'code':'vfx'},
                             {'id':2, 'type':'HumanUser', 'login':'luke.cole'},
                             {'id':3, 'type':'Project', 'sg_short_name':'hf2'},
                             {'id':4, 'type':'Shot'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = NoteTranslator(self.session)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_02_00
        """
        pass
    
    def test_translate(self):
        """
        Test that the translator converts the supplied test data as expected.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_03_00
            Remove obsolete tests for sg_note_type and sg_status_list, these are handled by valid_values on the property class now.
        .. versionchanged:: v00_04_00
            Update to use verify_translate() helper method in order to reduce code bloat.
        """
        verify_translate(self.translator, 'project', {'id':3, 'type':'Project'}, 'hf2', 'mm4')
        verify_translate(self.translator, 'user', {'id':2, 'type':'HumanUser'}, 'luke.cole', 'mark.streatfield')
        verify_translate(self.translator, 'note_links', [{'id':4, 'type':'Shot'}], [{'Shot':[['id', 'is', 4]]}], [{'Shot':[['id', 'is', 5]]}])
        verify_translate(self.translator, 'addressings_to', [{'id':1, 'type':'Group'}, {'id':2, 'type':'HumanUser'}], ['luke.cole', 'vfx'], ['anim'])
        verify_translate(self.translator, 'addressings_cc', [{'id':1, 'type':'Group'}, {'id':2, 'type':'HumanUser'}], ['luke.cole', 'vfx'], ['anim'])
        verify_translate(self.translator, 'custom_entity10_sg_notes_custom_entity10s', [{'id':4, 'type':'Shot'}], [{'Shot':[['id', 'is', 4]]}], [{'Shot':[['id', 'is', 2]]}])

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

