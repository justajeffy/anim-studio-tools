# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

from nose.tools import assert_equals, assert_raises, assert_true

from probe.fixtures.mock_tank import MockLabels, MockSystemObject, MockAssetObject, MockProperties, MockRevision

class TestMockLabels(object):
    """
    Nose unit test suite for Probe's Tank MockLabels.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        
        .. note::
            We just use strings to represent the label objects, for simplified testing code.
        """
        self.labels = {'LabelA': 'labelA', 'LabelB': 'labelB'}
        self.mock = MockLabels(self.labels)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass

    def test_init(self):
        """
        Verify that the mock labels object is initialised as expected.
        
        .. versionadded:: 0.2.0
        """
        assert_true(self.mock.labels == self.labels)
        
    def test_role_type(self):
        """
        Verify that the property style behaviour of labels works as expected (for RoleType labels).
        
        .. versionadded:: 0.2.0
        """
        self.mock.labels = {'RoleType': 'roleTypeA'}
        
        assert_true(self.mock.RoleType == 'roleTypeA')
        
class TestMockSystemObject(object):
    """
    Nose unit test suite for Probe's Tank MockSystemObject.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        self.mock = MockSystemObject('mock_name', 'mock_type')
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass

    def test_init(self):
        """
        Verify that the mock system object is initialised as expected.
        
        .. versionadded:: 0.2.0
        """
        assert_true(self.mock.name == 'mock_name')
        assert_true(self.mock.type_name == 'mock_type')
        
class TestMockAssetObject(object):
    """
    Nose unit test suite for Probe's Tank MockAssetObject.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        
        .. note::
            We just use strings to represent the label objects, for simplified testing code.
        """
        self.labels = {'LabelA': 'labelA', 'LabelB': 'labelB'}
        self.mock = MockAssetObject('mock_container', self.labels)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass

    def test_init(self):
        """
        Verify that the mock asset object is initialised as expected.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.12.0
            Added a simple test to confirm that the asset properties field is present.
        """
        assert_true(self.mock.container == 'mock_container')
        assert_true(self.mock.labels.labels == self.labels)
        assert_true(self.mock.system.name == 'test_name')
        assert_true(self.mock.system.type_name == 'test_type')
        assert_true(self.mock.properties != None)
        
    def test_save(self):
        """
        Verify that the mock revision has a save() method.
        
        .. versionadded:: 0.12.0
        """
        assert_true(hasattr(self.mock, 'save'))
        
    def test_repr(self):
        """
        Test that the mock asset object repr method works correctly.
        
        .. versionadded:: 0.2.0
        """
        assert_true(self.mock.__repr__() == 'mock_container')
        
class TestMockProperties(object):
    """
    Nose unit test suite for Probe's Tank MockProperties.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.7.0
            Added test data for is_locked property.
        
        .. note::
            We just use strings for property contents etc, in order to simplify unit test code.
        """
        self.properties = {'contents': ['RevisionA', 'RevisionB'], 'pipeline_data': {'cat': True, 'hat':False}, 'is_locked': False}
        self.mock = MockProperties(self.properties)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass

    def test_init(self):
        """
        Verify that the mock properties is initialised as expected.
        
        .. versionadded:: 0.2.0
        """
        assert_true(self.mock.properties == self.properties)
        
    def test_contents(self):
        """
        Test that the contents property returns the expected result.
        
        .. versionadded:: 0.2.0
        """
        assert_true(self.mock.contents == self.properties['contents'])
        
    def test_pipeline_data(self):
        """
        Test that the pipeline data property returns the expected result.
        
        .. versionadded:: 0.2.0
        """
        assert_true(self.mock.pipeline_data == self.properties['pipeline_data'])
        
    def test_is_locked(self):
        """
        Test that the is locked property returns the expected result.
        
        .. versionadded:: 0.7.0
        """
        assert_true(self.mock.is_locked == self.properties['is_locked'])
        
        # make sure we can set a value on it
        self.mock.is_locked = True
        assert_true(self.mock.is_locked)
        
    def test_keys(self):
        """
        Confirm that the expected property keys are returned.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.7.0
            Increment the number of expected keys by 1, added a check for is_locked.
        """
        assert_true(len(self.mock.keys()) == 3)
        
        assert_true('contents' in self.mock.keys())
        assert_true('pipeline_data' in self.mock.keys())
        assert_true('is_locked' in self.mock.keys())
        
class TestMockRevision(object):
    """
    Nose unit test suite for Probe's Tank MockRevision.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.12.0
            Added asset properties test data.
        
        .. note::
            We just use strings for revision properties in order to simplify unit test code.
        """
        self.revision = {
                         'asset': 'containerA',
                         'asset.labels': {'LabelA': 'labelA', 'LabelB': 'labelB'},
                         'asset.properties': {'is_locked':False},
                         'asset.system.type_name': 'ContainerA',
                         'system.name': 'revisionA',
                         'system.type_name': 'RevisionA',
                         'properties': {'pipeline_data':None},
                         }
        
        self.mock = MockRevision(self.revision)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass

    def test_init(self):
        """
        Verify that the mock revision is initialised as expected.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.12.0
            Confirm that the asset properties are correctly set.
        """
        assert_true(self.mock.asset.container == 'containerA')
        assert_true(self.mock.asset.properties.properties == self.revision['asset.properties'])
        assert_true(self.mock.asset.labels.labels == self.revision['asset.labels'])
        assert_true(self.mock.asset.system.type_name == 'ContainerA')
        assert_true(self.mock.system.name == 'revisionA')
        assert_true(self.mock.system.type_name == 'RevisionA')
        assert_true(self.mock.properties.properties == self.revision['properties'])

    def test_save(self):
        """
        Verify that the mock revision has a save() method.
        
        .. versionadded:: 0.7.0
        """
        assert_true(hasattr(self.mock, 'save'))

    def test_repr(self):
        """
        Test that the mock revision repr method works correctly.
        
        .. versionadded:: 0.2.0
        """
        assert_true(self.mock.__repr__() == 'containerA')

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

