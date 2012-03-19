# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

import os

from nose.tools import assert_true, assert_false

from probe.decorators.strace import strace

class TestStrace(object):
    """
    Nose unit test suite for Probe Stack Trace.
    
    .. versionadded:: 0.14.0
    """
    
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 0.14.0
        """
        
        files = ["/tmp/func_001_strace_results.strace"]
        
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 0.14.0
        """
        
        pass
    
    def test_001(self):
        """
        Test with follow children.
        
        .. versionadded:: 0.14.0
        """
        
        # Call the function
        func_001()
        
        # Check we have the required files
        assert_true(os.path.isfile("/tmp/func_001_strace_results.strace"))

@strace(follow_children=True, output_directory="/tmp")
def func_001():
    return

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

