#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from nose.tools import assert_equals, assert_raises, assert_true

from grenade.utils import status

class TestStatus(object):
    """
    Nose unit test suite for Grenade status module.
    
    .. versionadded:: 1.7.0
    """
    
    def test_001(self):
        """
        Test that the task status summary function is correctly working.
        
        .. versionadded:: 1.7.0
        .. versionchanged:: 1.7.4
            Updated unit test due to change in behaviour of get_pipeline_step_task_status_summary()
        """
        
        assert_equals(status.get_pipeline_step_task_status_summary([]), "na")
        assert_equals(status.get_pipeline_step_task_status_summary([""]), "na")
        assert_equals(status.get_pipeline_step_task_status_summary(["na"]), "na")
        assert_equals(status.get_pipeline_step_task_status_summary([None]), "na")
        assert_equals(status.get_pipeline_step_task_status_summary(["ip"]), "ip")
        assert_equals(status.get_pipeline_step_task_status_summary(["omt", "clsd"]), "inv")
        assert_equals(status.get_pipeline_step_task_status_summary(["inv", "wtg", "rdy"]), "inv")
        assert_equals(status.get_pipeline_step_task_status_summary(["conapr", "dap"]), "conapr")
        assert_equals(status.get_pipeline_step_task_status_summary(["hld", "wtg"]), "hld")
        assert_equals(status.get_pipeline_step_task_status_summary(["hld", "inv"]), "ip")

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

