#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_not_equals, assert_raises, assert_true

from ..fixtures.mock_translator import MockTranslator

from grenade.common.error import GrenadeModelCreateError, GrenadeModelReadError, GrenadeModelUpdateError, GrenadeModelDeleteError, \
    GrenadeModelPropertyReadOnlyError
from grenade.common.model import FIND_ONE, FIND_ALL, find, ModelBuilder, Model

from probe.fixtures.mock_shotgun import MockShotgun

class TestModelBuilder(object):
    """
    Nose unit test suite for Grenade ModelFactory.
    
    .. versionadded:: v00_03_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: v00_08_00
            Added a new field to the schema, to allow testing of default field values.
        .. versionchanged:: 0.9.0
            Added a new field to the schema, to allow testing of read-only field values.
        .. versionchanged:: 1.0.0
            Added a new schema field to allow testing that unsupported API data types are filtered out.
        """
        self.shotgun_schema = [{'MockModel': {'field_one':   {'data_type': {'value':'entity'}, 
                                                              'editable': {'value':True}, 
                                                              'properties': {'valid_values': {'value': ['MockModel']}}}, 
                                              'field_two':   {'data_type': {'value':'entity'}, 
                                                              'editable': {'value':True}, 
                                                              'properties': {'valid_values': {'value': ['MockModel']}}}, 
                                              'field_three': {'data_type': {'value':'entity'}, 
                                                              'editable': {'value':True}, 
                                                              'properties': {'valid_values': {'value': ['MockModel']}}},
                                              'field_four':  {'data_type': {'value':'string'}, 
                                                              'editable': {'value':False}, 
                                                              'properties': {'default_value': {'value': "default"}}},
                                              'field_pivot': {'data_type': {'value':'pivot_column'},
                                                              'properties': {}}}}]
        
        self.session = MockShotgun(schema=self.shotgun_schema, data=[])
        self.factory = ModelBuilder()
        
        self.translator = MockTranslator()

    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_03_00
        """
        pass
    
    def test_init(self):
        """
        Test that the model factory is initialised correctly.
        
        .. versionadded:: v00_05_01
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        factory = ModelBuilder(self.shotgun_schema)
        
        assert_equals(factory.schema, self.shotgun_schema)    # a pre-loaded schema is available
        assert_equals(self.factory.schema, None)              # no schema pre-loaded in this case
    
    def test_call(self):
        """
        Test that the model factory returns a correctly constructed model instance.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: v00_07_00
            Use the new property class accessors for the is_dirty field.
        .. versionchanged:: v00_08_00
            Test that setting a default value works as expected.
        .. versionchanged:: 0.9.0
            Test that read-only properties are supported as expected.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        .. versionchanged:: 1.0.0
            Update to ensure that unsupported API data types are automatically filtered out.
        """
        model = self.factory(self.session, 'MockModel', self.translator, field_one=1, field_two=2, field_three=3)
        
        assert_equals(model.identifier, 'MockModel')
        
        assert_equals(model['field_one'], '1')
        assert_equals(model.properties['field_one'].data_type, 'entity')
        assert_equals(model.properties['field_one'].get_is_dirty(), False)
        assert_equals(model.properties['field_one'].valid_values, ['MockModel'])
        assert_equals(model.properties['field_one'].is_read_only, False)
        
        assert_equals(model['field_two'], '2')
        assert_equals(model.properties['field_two'].data_type, 'entity')
        assert_equals(model.properties['field_two'].get_is_dirty(), False)
        assert_equals(model.properties['field_two'].valid_values, ['MockModel'])
        assert_equals(model.properties['field_two'].is_read_only, False)
        
        assert_equals(model['field_three'], 3)
        assert_equals(model.properties['field_three'].data_type, 'entity')
        assert_equals(model.properties['field_three'].get_is_dirty(), False)
        assert_equals(model.properties['field_three'].valid_values, ['MockModel'])
        assert_equals(model.properties['field_three'].is_read_only, False)
        
        assert_equals(model['field_four'], "default")
        assert_equals(model.properties['field_four'].data_type, 'string')
        assert_equals(model.properties['field_four'].get_is_dirty(), False)
        assert_equals(model.properties['field_four'].valid_values, [])
        assert_equals(model.properties['field_four'].is_read_only, True)
        
        assert_equals(model['id'], None)
        
        assert_true('field_pivot' not in model.properties.keys())

class TestModel(object):
    """
    Nose unit test suite for Grenade Model.
    
    .. versionadded:: v00_02_00
    .. versionchanged:: v00_03_00
        Pass model identifier into find calls, extended init() test to check new model data members.
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_03_00
            Updated due to changes to model constructor, add the MockModel schema to the MockShotgun instance.
        .. versionchanged:: 0.9.0
            Update schema in order to allow testing of read-only field behaviour.
        .. versionchanged:: 0.11.0
            Set the starting id for the mock shotgun id generator.
        .. versionchanged:: 1.0.0
            Set the datatype and properties schema entries on fields that were missing these values. Added a
            new schema field and test data to allow testing that unsupported API data types are filtered out.
        .. versionchanged:: 1.1.0
            Added Task schema definition to aid testing of never_editable fields.
        """
        self.shotgun_data = [{'id':1, 'type':'MockModel', 'field_one':1, 'field_two':2, 'field_three':3, 'field_pivot':'x'},
                             {'id':2, 'type':'MockModel', 'field_one':1, 'field_two':2, 'field_three':3},
                             {'id':3, 'type':'MockModel', 'field_one':3, 'field_two':2, 'field_three':1}]
        
        self.shotgun_schema = [{'MockModel': {'field_one':{'data_type': {'value':'integer'},
                                                           'properties': {}}, 
                                              'field_two':{'data_type': {'value':'integer'},
                                                           'properties': {}}, 
                                              'field_three':{'data_type': {'value':'integer'}, 
                                                             'editable': {'value':False}, 
                                                             'properties': {'default_value': {'value': 99}}}},
                                              'field_pivot':{'data_type': {'value':'pivot_column'}}},
                               {'Task': {'field_one': {}, 'dependency_violation': {'data_type': {'value':'boolean'}, 
                                                                                   'editable': {'value':False},
                                                                                   'properties':{'default_value':{'value':True}}}}}]

        self.session = MockShotgun(schema=self.shotgun_schema, data=self.shotgun_data, first_id=4)
        self.identifier = 'MockModel'
        self.translator = MockTranslator(self.session)
        
        self.model_schema = {'field_one': None, 'field_two': None, 'field_three': None}
        
        # manually build a model for this test
        self.model = Model(self.session, self.identifier, self.translator, self.model_schema)
        
        self.model['field_one'] = 1
        self.model['field_two'] = 2
        self.model['field_three'] = 3
        
        self.model._purify()
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_02_00
        """
        pass
    
    def test_init(self):
        """
        Test that the test model has been correctly initialised (all required fields present, purified, etc).
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_03_00
            Added extra checks for model identifier and schema.
        .. versionchanged:: v00_07_00
            Use the new property class accessors for the is_dirty field.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        # make sure the session, identifier and translator members are valid
        assert_equals(self.model.session, self.session)
        assert_equals(self.model.identifier, self.identifier)
        assert_equals(self.model.translator, self.translator)

        schema = []
        for key in self.model_schema.keys():
            schema.append(key)
        schema.append('id')

        assert_equals(self.model.__schema__, schema)
        
        # make sure the id is present, pure, but nullified
        assert_true('id' in self.model.properties.keys())
        assert_equals(self.model.properties['id'].value, None)
        assert_equals(self.model.properties['id'].get_is_dirty(), False)
        
        # make sure a property exists for each entry in the schema
        for entry in self.model_schema:
            assert_true(entry in self.model.properties.keys())
        
        # make sure that the properties were correctly initialised
        for key in self.model_schema:
            assert_not_equals(self.model.properties[key].value, None)
            assert_equals(self.model.properties[key].get_is_dirty(), False)
            
    def test_find(self):
        """
        Test that the find model module method behaves correctly, in both FIND_ONE and FIND_ALL modes.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_08_00
            Implemented support for the optional sorting argument.
        .. versionchanged:: v00_08_03
            Implement support for the no results error checking functionality when using FIND_ONE mode.
        .. versionchanged:: 0.10.0
            Added tests for the new 'fields' key work argument.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        .. versionchanged:: 1.0.0
            Update to ensure that unsupported API data types are automatically filtered out.
        """
        results = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        
        assert_not_equals(results, None)
        assert_equals(results['id'], 1)                             # make sure we received the entity we expected
        assert_true('field_pivot' not in results.properties.keys()) # make sure unsupported data types are filtered out
        
        results = find(self.session, 'MockModel', [['field_one', 'is', 1]], [{'field_name': 'id', 'direction': 'desc'}], FIND_ALL)
        
        assert_not_equals(results, [])
        assert_equals(len(results), 2)
        
        for result in results:
            assert_not_equals(result['id'], 3)    # make sure we only received entities that match
            
        # verify the results were sorted as expected
        assert_equals(results[0]['id'], 2)
        assert_equals(results[1]['id'], 1)
        
        # check that an error is thrown when no entities for the given filters could be found
        assert_raises(GrenadeModelCreateError, find, self.session, 'MockModel', [['id', 'is', 0]], [], FIND_ONE)
        
        results = find(self.session, 'MockModel', [['id', 'is', 1]], order=[], mode=FIND_ONE, fields=["field_three"])
        
        assert_not_equals(results, None)
        assert_equals(results['field_three'], 3)
        
        assert_raises(KeyError, results.get_property, 'field_one')
    
    def test_get_property(self):
        """
        Test that the correct value is returned when getting a property. Explicit calls to get_property()
        as well as the __getitem__ mode are included.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        # make sure we receive the correct (translated) value where relevant
        assert_equals(self.model.get_property('field_one'), '1')
        assert_equals(self.model.get_property('field_two'), '2')
        assert_equals(self.model.get_property('field_three'), 3)
        
        # make sure that __getitem__ mode works in exactly the same way
        assert_equals(self.model['field_one'], '1')
        assert_equals(self.model['field_two'], '2')
        assert_equals(self.model['field_three'], 3)
    
    def test_set_property(self):
        """
        Test that the correct value is applied when setting a property. Explicit calls to set_property()
        as well as the __setitem__ mode are included.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_07_00
            Use the new property class accessors for the is_dirty field.
        .. versionchanged:: 0.9.0
            Verify that it is not possible to set the value of a read-only field.
        .. versionchanged:: 0.9.1
            Verify that it is possible to set the value of a read-only field on an uncreated model.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        .. versionchanged:: 0.11.0
            Update broken identifier (use MockModel instead of Note).
        """
        # make sure we store the correct (translated) value where relevant
        self.model.set_property('field_one', 'one')
        self.model.set_property('field_three', 'three')
        
        assert_equals(self.model.properties['field_one'].value, 'one')
        assert_equals(self.model.properties['field_one'].get_is_dirty(), True)
        assert_equals(self.model.properties['field_three'].value, 'three')
        assert_equals(self.model.properties['field_three'].get_is_dirty(), True)
        
        # make sure that __setitem__ mode works in the same way
        self.model['field_one'] = 1
        self.model['field_three'] = 3
        
        assert_equals(self.model.properties['field_one'].value, '1')
        assert_equals(self.model.properties['field_one'].get_is_dirty(), True)
        assert_equals(self.model.properties['field_three'].value, 3)
        assert_equals(self.model.properties['field_three'].get_is_dirty(), True)
        
        # make sure we can't assign a value to a read-only property
        model = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        
        assert_raises(GrenadeModelPropertyReadOnlyError, model.set_property, 'field_three', 'three')
        
        # make sure we can force assign a value to a read-only property
        model.set_property('field_three', 'three', force=True)
        assert_equals(model.properties['field_three'].value, 'three')
        
        # make sure we can assign a value to a read-only property in an uncreated model
        model = ModelBuilder()(self.session, 'MockModel', None)
        model.set_property('field_three', 'three')
        assert_equals(model.properties['field_three'].value,'three')
        
    def test_create(self):
        """
        Test model creation behaviour.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: 0.9.0
            Verify that created models have all read-only fields, etc populated with
            values from Shotgun.
        .. versionchanged:: 0.9.1
            Verify that it is possible to set read-only fields on models at creation time.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        .. versionchanged:: 1.1.0
            Update to confirm that never_editable fields are ignored during creation.
        """
        # step 1: create a non-existant item
        result = self.model.create()
        assert_not_equals(result['id'], None)
        
        # step 2: attempt to recreate an existing item
        assert_raises(GrenadeModelCreateError, self.model.create)
        
        # step 3: verify that all fields are populated
        model = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        model.properties['id'].value = None          # cheat
        model.properties['field_three'].value = None # cheat
        
        model.create() # populates read-only fields, etc from shotgun
        
        assert_equals(model['field_three'], 99)
        
        # step 4: verify that we can set a read-only field at creation time
        model = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        model.properties['id'].value = None
        model['field_three'] = 3
        
        model.create() # sets read-only fields to specified creation value
        
        assert_equals(model['field_three'], 3)
        
        # step 5: verify that never_editable fields are ignored at creation time
        model = Model(self.session, 'Task', None, self.shotgun_schema[1]['Task'])
        model['field_one'] = 1
        model['dependency_violation'] = False
        assert_true(model.properties.has_key('dependency_violation'))
        
        result = model.create() # should skip dependency_violation field
        
        assert_not_equals(result['id'], None)
        assert_equals(result['dependency_violation'], True) # confirm we got default
    
    def test_read(self):
        """
        Test model read behaviour.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_08_00
            Updated find() calls due to addition of order by argument.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        # step 1: select an existing item
        model_orig = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        model_copy = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        
        # step 2: apply a change
        model_orig['field_two'] = 'two'
        model_orig.update()
        
        # step 3: verify the change is read back in
        assert_equals(model_copy['field_two'], 2)
        model_copy.read()
        assert_equals(model_copy['field_two'], 'two')
        
        # step 4: try to read a non-existent item
        model_fail = self.model
        model_fail['id'] = 5
        model_fail.read()
        
        for key in model_fail.properties.keys():
            assert_equals(model_fail[key], None)
            
        # step 5: try to read a broken item
        assert_raises(GrenadeModelReadError, self.model.read)
        
    def test_update(self):
        """
        Test model update behavior.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_08_00
            Updated find() calls due to addition of order by argument.
        .. versionchanged:: 0.9.0
            Verify that read-only fields aren't passed in to shotgun during an update.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        # step 1: select an existing item
        model = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        
        # step 2: apply a change
        model['field_two'] = 'two'
        result = model.update()
        
        assert_equals(len(result), 1)
        assert_true('field_two' in result)
        
        # step 3: try to update a broken item
        assert_raises(GrenadeModelUpdateError, self.model.update)
        
        # step 4: try to update a read-only item
        model = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        
        model['field_two'] = 'two'
        model.properties['field_two'].is_read_only = True   # cheat
        
        result = model.update()
        assert_equals(result, [])
    
    def test_delete(self):
        """
        Test model delete behavior.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_08_00
            Updated find() calls due to addition of order by argument.
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        # step 1: select an existing item
        model = find(self.session, 'MockModel', [['id', 'is', 1]], [], FIND_ONE)
        
        # step 2: delete it
        result = model.delete()
        
        assert_equals(result, True)
        for key in model.properties.keys():
            assert_equals(model[key], None)
            
        # step 3: try to delete a non-existent item
        model_fail = self.model
        model_fail['id'] = 5
        result = model_fail.delete()
        
        assert_equals(result, False)
            
        # step 4: try to delete a broken item
        self.model['id'] = None
        assert_raises(GrenadeModelDeleteError, self.model.delete)
    
    def test_repr(self):
        """
        Test that the model representation is generated as expected.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        result = self.model.__repr__()

        assert_equals(result, "{'field_one': '1', 'field_two': '2', 'field_three': 3, 'id': None}")

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

