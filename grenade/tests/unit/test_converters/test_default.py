#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_not_equals, assert_raises, assert_true

from datetime import datetime

from grenade.common.error import GrenadeValidationError
from grenade.converters.default import convert_asset, convert_datetime, convert_department, convert_group, convert_imgseqs, \
    convert_link, convert_links, convert_meetings, convert_project, convert_recipients, convert_scene, convert_shot, \
    convert_sequence, convert_step, convert_user
    
from probe.fixtures.mock_shotgun import MockShotgun

class TestDefaultConverters(object):
    """
    Nose unit test suite for Grenade default converters.
    
    .. versionadded:: v00_03_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_03_00
        """
        self.shotgun_data = [{'id':1, 'type':'Group', 'code':'vfx'},
                             {'id':2, 'type':'HumanUser', 'login':'luke.cole'},
                             {'id':3, 'type':'Project', 'sg_short_name':'hf2'},
                             {'id':4, 'type':'Shot', 'code':'19d_010', 'project':{'id':3, 'type':'Project'}},
                             {'id':5, 'type':'Scene', 'code':'19d', 'project':{'id':3, 'type':'Project'}},
                             {'id':6, 'type':'Step', 'code':'Anim', 'entity_type':'Shot'},
                             {'id':7, 'type':'Asset', 'code':'mumble'},
                             {'id':8, 'type':'Sequence', 'code':'test'},
                             {'id':9, 'type':'CustomNonProjectEntity03', 'code':'anim'},
                             {'id':10, 'type':'Version', 'code':'900_001_anim_v001'},
                             {'id':11, 'type':'CustomEntity01', 'code':'Important Meeting'}]

        self.session = MockShotgun(schema=[], data=self.shotgun_data)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_03_00
        """
        pass
    
    def test_convert_asset(self):
        """
        Test that the asset converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_04_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_asset(self.session, 'mumble')
        assert_equals(result, {'id':7, 'type':'Asset'})
        assert_raises(GrenadeValidationError, convert_asset, self.session, 'gloria')
            
    def test_convert_datetime(self):
        """
        Test that the datetime converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_05_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_datetime(self.session, '30/08/2010 15:00')
        assert_equals(result, datetime(2010, 8, 30, 15, 0))
    
    def test_convert_department(self):
        """
        Test that the department converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_05_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_department(self.session, 'anim')
        assert_equals(result, {'id':9, 'type':'CustomNonProjectEntity03'})
        assert_raises(GrenadeValidationError, convert_department, self.session, 'fx')

    def test_convert_group(self):
        """
        Test that the group converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_04_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_group(self.session, 'vfx')
        assert_equals(result, {'id':1, 'type':'Group'})
        assert_raises(GrenadeValidationError, convert_group, self.session, 'anim')
    
    def test_convert_imgseqs(self):
        """
        Test that the image sequences converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_05_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_imgseqs(self.session, ['900_001_anim_v001'])
        assert_equals(len(result), 1)
        for entity in result:
            assert_true(entity in [{'id':10, 'type':'Version'}])
        assert_raises(GrenadeValidationError, convert_imgseqs, self.session, ['900_002_fx_v009'])

    def test_convert_link(self):
        """
        Test that the link converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_06_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_link(self.session, {'Shot':[['id', 'is', 4]]})
        assert_equals(result, {'id':4, 'type':'Shot'})
        assert_raises(GrenadeValidationError, convert_link, self.session, {'Shot':[['id', 'is', 5]]})
    
    def test_convert_links(self):
        """
        Test that the links converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: 0.9.0
            Check that an exception is raised when invalid links data is provided.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_links(self.session, [{'Shot':[['id', 'is', 4]]}])
        assert_equals(result, [{'id':4, 'type':'Shot'}])
        assert_raises(GrenadeValidationError, convert_links, self.session, [{'Shot':[['id', 'is', 5]]}])
        assert_raises(GrenadeValidationError, convert_links, self.session, {'Shot':[['id', 'is', 4]]})
            
    def test_convert_meetings(self):
        """
        Test that the meetings converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_05_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_meetings(self.session, ['Important Meeting'])
        assert_equals(len(result), 1)
        for entity in result:
            assert_true(entity in [{'id':11, 'type':'CustomEntity01'}])
        assert_raises(GrenadeValidationError, convert_meetings, self.session, ['Boring Meeting'])
    
    def test_convert_project(self):
        """
        Test that the project converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_project(self.session, 'hf2')
        assert_equals(result, {'id':3, 'type':'Project'})
        assert_raises(GrenadeValidationError, convert_project, self.session, 'mm4')

    def test_convert_recipients(self):
        """
        Test that the recipients converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_recipients(self.session, ['luke.cole', 'vfx'])
        assert_equals(len(result), 2)
        for entity in result:
            assert_true(entity in [{'id':1, 'type':'Group'}, {'id':2, 'type':'HumanUser'}])
        assert_raises(GrenadeValidationError, convert_recipients, self.session, ['anim'])
            
    def test_convert_scene(self):
        """
        Test that the scene converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_04_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_scene(self.session, 'hf2:19d')
        assert_equals(result, {'id':5, 'type':'Scene'})
        assert_raises(GrenadeValidationError, convert_scene, self.session, 'hf2:20a')
        assert_raises(GrenadeValidationError, convert_scene, self.session, 'hf2:20a:123')

    def test_convert_sequence(self):
        """
        Test that the sequence converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_04_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_sequence(self.session, 'test')
        assert_equals(result, {'id':8, 'type':'Sequence'})
        assert_raises(GrenadeValidationError, convert_sequence, self.session, 'none')

    def test_convert_shot(self):
        """
        Test that the scene converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_04_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_shot(self.session, 'hf2:19d_010')
        assert_equals(result, {'id':4, 'type':'Shot'})
        assert_raises(GrenadeValidationError, convert_shot, self.session, 'hf2:20a_010')
        assert_raises(GrenadeValidationError, convert_shot, self.session, 'hf2:20a_010:123')

    def test_convert_step(self):
        """
        Test that the step converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_04_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        .. versionchanged:: 1.2.0
            Update due to change in step specification (entity type now needed).
        """
        result = convert_step(self.session, 'Shot:Anim')
        assert_equals(result, {'id':6, 'type':'Step'})
        assert_raises(GrenadeValidationError, convert_step, self.session, 'Comp')
    
    def test_convert_user(self):
        """
        Test that the user converter transforms the supplied test data correctly.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = convert_user(self.session, 'luke.cole')
        assert_equals(result, {'id':2, 'type':'HumanUser'})
        assert_raises(GrenadeValidationError, convert_user, self.session, 'mark.streatfield')

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

