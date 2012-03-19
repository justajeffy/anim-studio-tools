# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

from nose.tools import assert_equals, assert_true

from probe.fixtures.mock_log import MockLog

class TestMockLog(object):
    """
    Nose unit test suite for Probe MockLog.
    
    .. versionadded:: 0.2.0
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        self.log = MockLog()
        self.message = 'test_message'
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.2.0
        """
        pass
    
    def test_init(self):
        """
        Test that the mock log object is initialised correctly.
        
        .. versionadded:: 0.2.0
        """
        # make sure the required message logs are present
        assert_true(self.log.messages.has_key('info'))
        assert_true(self.log.messages.has_key('error'))
        assert_true(self.log.messages.has_key('exception'))
        assert_true(self.log.messages.has_key('debug'))
        assert_true(self.log.messages.has_key('warn'))

        # make sure all the message logs are empty!
        assert_equals(self.log.messages['info'], [])
        assert_equals(self.log.messages['error'], [])
        assert_equals(self.log.messages['exception'], [])
        assert_equals(self.log.messages['debug'], [])
        assert_equals(self.log.messages['warn'], [])
        
    def test_info(self):
        """
        Test that the mock log object correctly records an info message.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.log.messages['info'], [])
        self.log.info(self.message)
        assert_equals(self.log.messages['info'], [self.message])

        # the other message logs should be empty
        assert_equals(self.log.messages['error'], [])
        assert_equals(self.log.messages['exception'], [])
        assert_equals(self.log.messages['debug'], [])
        assert_equals(self.log.messages['warn'], [])
        
    def test_error(self):
        """
        Test that the mock log object correctly records an error message.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.log.messages['error'], [])
        self.log.error(self.message)
        assert_equals(self.log.messages['error'], [self.message])

        # the other message logs should be empty
        assert_equals(self.log.messages['info'], [])
        assert_equals(self.log.messages['exception'], [])
        assert_equals(self.log.messages['debug'], [])
        assert_equals(self.log.messages['warn'], [])
        
    def test_exception(self):
        """
        Test that the mock log object correctly records an exception message.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.log.messages['exception'], [])
        self.log.exception(self.message)
        assert_equals(self.log.messages['exception'], [self.message])

        # the other message logs should be empty
        assert_equals(self.log.messages['info'], [])
        assert_equals(self.log.messages['error'], [])
        assert_equals(self.log.messages['debug'], [])
        assert_equals(self.log.messages['warn'], [])
        
    def test_debug(self):
        """
        Test that the mock log object correctly records a debug message.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.log.messages['debug'], [])
        self.log.debug(self.message)
        assert_equals(self.log.messages['debug'], [self.message])

        # the other message logs should be empty
        assert_equals(self.log.messages['info'], [])
        assert_equals(self.log.messages['error'], [])
        assert_equals(self.log.messages['exception'], [])
        assert_equals(self.log.messages['warn'], [])
        
    def test_warn(self):
        """
        Test that the mock log object correctly records a warning message.
        
        .. versionadded:: 0.2.0
        """
        assert_equals(self.log.messages['warn'], [])
        self.log.warn(self.message)
        assert_equals(self.log.messages['warn'], [self.message])

        # the other message logs should be empty
        assert_equals(self.log.messages['info'], [])
        assert_equals(self.log.messages['error'], [])
        assert_equals(self.log.messages['exception'], [])
        assert_equals(self.log.messages['debug'], [])
        
    def test_clear(self):
        """
        Test that the clear() method correctly empties the mock log instance.
        
        .. versionadded:: 0.2.0
        """
        self.log.info(self.message)
        self.log.error(self.message)
        self.log.exception(self.message)
        self.log.debug(self.message)
        self.log.warn(self.message)
        
        self.log.clear()
        
        # make sure all the message logs are empty!
        assert_equals(self.log.messages['info'], [])
        assert_equals(self.log.messages['error'], [])
        assert_equals(self.log.messages['exception'], [])
        assert_equals(self.log.messages['debug'], [])
        assert_equals(self.log.messages['warn'], [])

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

