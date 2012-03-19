# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

from nose.tools import assert_equals, assert_raises, assert_true

from probe.fixtures.mock_database import MockDatabase, MockDatabaseError, MockDatabaseFilter, MockResultProxy

class TestMockDatabaseError(object):
    """
    Nose unit test suite for Probe MockDatabaseError.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        self.valid_errors = ['MockDatabaseError']
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass

    def test_error(self):
        """
        Verify that the mock_database module contains the required error symbols.
        
        .. versionadded:: 0.2.0
        """
        import probe.fixtures.mock_database
        
        # pretty lame, but not much else we can do
        for error in self.valid_errors:
            assert_true(error in dir(probe.fixtures.mock_database))
            
class TestMockResultProxy(object):
    """
    Nose unit test suite for Probe MockResultProxy.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        self.proxy = MockResultProxy([1, 2, 3])
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass
    
    def test_init(self):
        """
        Verify the result proxy has been initialised correctly.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.proxy.data, [1, 2, 3])
    
    def test_first(self):
        """
        Test that the first result in the proxy is returned.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.3.0
            Changed to check for None on call to first() when empty.
        """
        assert_equals(self.proxy.first(), 1)
        
        # test that the behavior is correct when we're working with an empty proxy
        self.empty = MockResultProxy()
        assert_equals(self.empty.first(), None)
    
    def test_options(self):
        """
        Test that calling options just returns self.
        
        .. versionadded:: 0.3.0
        """
        assert_equals(self.proxy.options(), self.proxy)
        assert_equals(self.proxy.options(1,"random",3), self.proxy)
    
    def test_one(self):
        """
        Test that extracting one result from the proxy works as expected.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.3.0
            Changed to check for None on call to one() when empty.
        """
        assert_equals(self.proxy.one(), 1)
        
        # test that the behavior is correct when we're working with an empty proxy
        self.empty = MockResultProxy()
        assert_equals(self.empty.one(), None)
        
    def test_all(self):
        """
        Test that all of the results in the proxy are returned.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.proxy.all(), [1, 2, 3])
    
class TestMockDatabaseFilter(object):
    """
    Nose unit test suite for Probe MockDatabaseFilter.
    
    .. versionadded:: 0.2.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        self.parser = MockDatabaseFilter([1, 2, 3])
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass
    
    def test_init(self):
        """
        Verify the result filter has been initialised correctly.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.parser.data, [1, 2, 3])
        
    def test_filter(self):
        """
        Test that the results are filtered as expected (i.e., not at all)
        
        .. versionadded:: 0.2.0
        """
        result = self.parser.filter(None)
        
        assert_equals(type(result), MockResultProxy)
        assert_equals(result.data, self.parser.data)
        
    def test_all(self):
        """
        Test that all of the results in the internal proxy are returned.
        
        .. versionadded:: 0.14.5
        """
        assert_equals(self.parser.all(), [1, 2, 3])
    
class TestMockDatabase(object):
    """
    Nose unit test suite for Probe MockDatabase.
    
    .. versionadded:: 0.2.0
    """
    
    class MockRecord:
        def __init__(self, value):
            self.id = None
            self.value = value
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        self.database = MockDatabase()
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass
    
    def test_init(self):
        """
        Test that the database has been initialised correctly.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.database.pending, [])
        assert_equals(self.database.data, [])
        assert_equals(self.database.unique, False)
        
        assert_true(self.database.keys)
        
    def test_add(self):
        """
        Test that data has been added to the database.
        
        .. versionadded:: 0.2.0
        """
        self.database.add("data")
        
        assert_equals(self.database.pending, ["data"])
        assert_equals(self.database.data, [])
        
    def test_commit(self):
        """
        Test that changes have been comitted to the database.
        
        .. versionadded:: 0.2.0
        """
        record = self.MockRecord("data")
        
        self.database.add(record)
        
        assert_equals(self.database.pending, [record])
        assert_equals(self.database.data, [])
        assert_equals(self.database.pending[0].id, None)
        
        self.database.commit()
        
        assert_equals(self.database.pending, [])
        assert_equals(self.database.data, [record])
        assert_true(self.database.data[0].id != None)
        
        # check that the behaviour is "correct" when unique mode is enabled
        self.database.unique = True
        self.database.add(record)
        
        assert_equals(self.database.pending, [record])
        assert_equals(self.database.data, [record])
        
        assert_raises(MockDatabaseError, self.database.commit)
    
    def test_query(self):
        """
        Test that the queried results are returned from the database.
        
        .. versionadded:: 0.2.0
        """
        record = self.MockRecord("data")                    # setup some test data
        
        self.database.add(record)
        self.database.commit()
        
        result = self.database.query(self.MockRecord)       # actually do the test
        
        assert_equals(type(result), MockDatabaseFilter)     # make sure the expected data type is returned
        assert_equals(result.filter(None).all(), [record])  # make sure the query actually finds the record
    
    def test_rollback(self):
        """
        Test that pending changes are rolled back.
        
        .. versionadded:: 0.2.0
        """
        self.database.add("data")
        
        assert_equals(self.database.pending, ["data"])
        assert_equals(self.database.data, [])
        
        self.database.rollback()
        
        assert_equals(self.database.pending, [])
        assert_equals(self.database.data, [])

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

