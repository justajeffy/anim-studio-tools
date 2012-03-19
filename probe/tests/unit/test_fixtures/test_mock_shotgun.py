# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

from nose.tools import assert_equals, assert_true

from probe.fixtures.mock_shotgun import MockShotgun

class TestMockShotgun(object):
    """
    Nose unit test suite for Probe MockShotgun.
    
    .. versionadded:: 0.2.0
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        self.session = MockShotgun()
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass
    
    def test_init(self):
        """
        Verify that the mock shotgun object is correctly initialised.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.3.0
            Ensure the base_url is correctly initialised.
        """
        assert_equals(self.session.host, '')
        assert_equals(self.session.user, '')
        assert_equals(self.session.skey, '')
        assert_equals(self.session.base_url, self.session.host)
        assert_equals(self.session.schema, [])
        assert_equals(self.session.data, [])
        
    def test_find(self):
        """
        Test the mock shotgun object's find() method!
        
        .. versionadded:: 0.2.0
        """
        # inject some test data
        e1 = self.session.create('MockEntity', {'field':'test_value'})
        e2 = self.session.create('MockEntity', {'field':'test_value'})
        e3 = self.session.create('MockEntity', {'field':'different_value'})
        
        assert_true(len(self.session.data) == 3)
        
        # perform the actual test
        results = self.session.find('MockEntity', [['field', 'is', 'test_value']])
        
        assert_true(len(results) == 2)
        for result in results:
            assert_true(result['id'] in [e1['id'], e2['id']])
    
    def test_find_one(self):
        """
        Test the mock shotgun object's find_one() method!
        
        .. versionadded:: 0.2.0
        """
        # inject some test data
        e1 = self.session.create('MockEntity', {'field':'test_value'})
        e2 = self.session.create('MockEntity', {'field':'test_value'})
        e3 = self.session.create('MockEntity', {'field':'different_value'})
        
        assert_true(len(self.session.data) == 3)
        
        # perform the actual test(s)
        result = self.session.find_one('MockEntity', [['field', 'is', 'test_value']])
        assert_true(result['id'] == e1['id'])
        
        result = self.session.find_one('MockEntity', [['field', 'is', 'test_foobar']])
        assert_true(result == None)
    
    def test_create(self):
        """
        Check that new entities are created within the mock shotgun object as expected.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.3.0
            Updated to support/test return fields, id generation, etc.
        .. versionchanged:: 0.8.0
            Test that the correct id is generated on create() when first_id has been set
            on the mock shotgun object.
        .. versionchanged:: 0.8.0
            Test that the default value auto-population behaviour is working correctly.
        """
        data = self.session.create('MockEntity', {'field':'test_value'}, return_fields=['field'])
        
        assert_true(len(self.session.data) == 1)
        assert_true(self.session.data == [data])
        assert_true(data.has_key('field'))
        
        assert_true(self.session.data[0].has_key('id'))
        assert_true(self.session.data[0]['id'], 1)
        
        assert_true(self.session.data[0].has_key('type'))
        assert_true(self.session.data[0]['type'] == 'MockEntity')
        
        assert_true(self.session.data[0].has_key('field'))
        assert_true(self.session.data[0]['field'] == 'test_value')
        
        # test the special shot entity hack
        data = self.session.create('Shot', {'code':'test_shot'}, return_fields=['sg_status_list'])
        
        assert_true(data.has_key('sg_status_list'))
        assert_true(data['sg_status_list'] == '')
        
        # test the id generation behaviour
        self.session = MockShotgun(first_id=2)
        data = self.session.create('MockEntity', {'field':'test_value'}, return_fields=['field'])

        assert_true(data['id'] == 2)
        
        # test the default value auto-population behaviour
        schema = [{'MockEntity': {'field_one':{}, 'field_two':{'data_type': {'value':'integer'}, 
                                                               'editable': {'value':False}, 
                                                               'properties': {'default_value': {'value': 99}}}}}]
        self.session = MockShotgun(schema=schema)
        data = self.session.create('MockEntity', {'field_one':'test_value'}, return_fields=['field_one', 'field_two'])
        
        assert_true(data['field_two'], 99)
    
    def test_update(self):
        """
        Check that the specified entity is updated as expected.
        
        .. versionadded:: 0.2.0
        """
        # inject some test data
        data = self.session.create('MockEntity', {'field':'test_value'})

        assert_true(len(self.session.data) == 1)
        assert_true(self.session.data[0]['field'] == 'test_value')
        
        # perform the actual test
        self.session.update('MockEntity', data['id'], {'field':'updated'})
        
        assert_true(len(self.session.data) == 1)
        assert_true(self.session.data[0]['field'] == 'updated')
    
    def test_delete(self):
        """
        Check that entities are deleted from the mock shotgun object as expected.
        
        .. versionadded:: 0.2.0
        """
        # inject some test data
        data = self.session.create('MockEntity', {'field':'test_value'})
        
        assert_true(len(self.session.data) == 1)
        assert_true(self.session.data[0]['field'] == 'test_value')
        
        # perform the actual test
        self.session.delete('MockEntity', data['id'])
        
        assert_true(len(self.session.data) == 0)
        
    def test_upload_thumbnail(self):
        """
        Check that it is possible to upload thumbnails to the mock shotgun object.
        
        .. versionadded:: 0.3.0
        """
        assert_true(hasattr(self.session, 'upload_thumbnail'))
        
        self.session.upload_thumbnail('MockEntity', 1, '/drd/transfer/fake_thumbnail.jpg')
        
        assert_true(True) # we can't actually check anything, as the method does nothing ...
    
    def test_schema_field_read(self):
        """
        Check that the schema_field_read() method works as expected.
        
        .. versionadded:: 0.2.0
        """
        schema = {'MockEntity': {'field':   {'data_type': {'value':'entity'}, 'properties': {'valid_values': {'value': ['TestEntity']}}}}} 
        self.session.schema = [schema]
        
        assert_true(self.session.schema_field_read('MockEntity') == schema['MockEntity'])
        assert_true(self.session.schema_field_read('MockEntity', 'field')['field'] == schema['MockEntity']['field'])

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

