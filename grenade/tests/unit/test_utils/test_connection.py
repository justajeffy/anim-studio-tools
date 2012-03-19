#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_raises, assert_true

from grenade.common.error import GrenadeConnectionError
from grenade.utils.connection import Connection

from probe.fixtures.mock_log import MockLog

class TestConnection(object):
    """
    Nose unit test suite for Grenade Shotgun Connection wrapper.
    
    .. versionadded:: v00_03_00
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: v00_03_00
        """
        self.connection = Connection(host='test_host', user='test_user', skey='test_skey')
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: v00_03_00
        """
        pass
    
    def test_init(self):
        """
        Test that the Connection object is correctly initialised.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        assert_equals(self.connection.host, 'test_host')
        assert_equals(self.connection.user, 'test_user')
        assert_equals(self.connection.skey, 'test_skey')
        
        assert_equals(self.connection.session, None)
    
    def test_connect(self):
        """
        Test that connect() behaves as expected.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        log = MockLog()

        try:
            session = self.connection.connect(log)
            assert_true(False)    # we shouldn't ever get here
        except GrenadeConnectionError, e:
            assert_equals(len(log.messages['error']), 1)
            assert_true('Unable to connect' in log.messages['error'][0])
        
    def test_get_session(self):
        """
        Test that the Connection.get_session() method behaves correctly.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: 0.11.0
            Update to use nose asserts statements.
        """
        log = MockLog()
        
        # should get None on an unconnected connection
        session = self.connection.get_session()
        assert_equals(session, None)
        
        # should also get None on an invalid connection
        try:
            self.connection.connect(log)
            assert_true(False)    # we shouldn't ever get here
        except GrenadeConnectionError, e:
            session = self.connection.get_session()
            assert_equals(session, None)

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

