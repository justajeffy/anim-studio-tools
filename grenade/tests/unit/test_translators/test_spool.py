#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..helpers.translators import verify_translate

from grenade.translators.spool import SpoolTranslator
from probe.fixtures.mock_shotgun import MockShotgun

from datetime import datetime

class TestShotTranslator(object):
    """
    Nose unit test suite for Grenade ShotTranslator.
    
    .. versionadded:: 1.7.2
    """
    def setup(self):
        """
        Set up the unit test suite.
        
        .. versionadded:: 1.7.2
        """
        self.shotgun_data = [{'id':1, 'type':'HumanUser', 'login':'stephen.beeson'}]
        
        self.session = MockShotgun(schema=[], data=self.shotgun_data)
        self.translator = SpoolTranslator(self.session)
    
    def teardown(self):
        """
        Tear down the unit test suite.
        
        .. versionadded:: 1.7.2
        """
        pass
    
    def test_translate(self):
        """
        Test that the translator converts the supplied test data as expected.
        
        .. versionadded:: 1.7.2
        """
        verify_translate(self.translator, 'sg_checksum_qa', [{'type': 'HumanUser', 'id': 1}], ['stephen.beeson'], ['hugh.raynor'])
        verify_translate(self.translator, 'sg_delivery_time', datetime(2010, 8, 30, 15, 0), '30/08/2010 15:00', '30-08-2010 15.00')
        verify_translate(self.translator, 'sg_edit_qa', [{'type': 'HumanUser', 'id': 1}], ['stephen.beeson'], ['hugh.raynor'])
        verify_translate(self.translator, 'sg_delivered_by', {'type': 'HumanUser', 'id': 1}, 'stephen.beeson', 'hugh.raynor')
        verify_translate(self.translator, 'sg_tank_created_by', {'type': 'HumanUser', 'id': 1}, 'stephen.beeson', 'hugh.raynor')
        verify_translate(self.translator, 'sg_tank_creation_date', datetime(2010, 8, 30, 15, 0), '30/08/2010 15:00', '30-08-2010 15.00')
        verify_translate(self.translator, 'sg_visual_qa', [{'type': 'HumanUser', 'id': 1}], ['stephen.beeson'], ['hugh.raynor'])
        verify_translate(self.translator, 'sg_notes', [{'id':1, 'type':'HumanUser'}], [{'HumanUser':[['id', 'is', 1]]}], [{'HumanUser':[['id', 'is', 2]]}])

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

