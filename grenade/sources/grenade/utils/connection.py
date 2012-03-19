#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

import os

from shotgun_v3 import Shotgun

from ..common.error import GrenadeConnectionError

class Connection(object):
    """
    Grenade shotgun connection.
    
    .. note::
        Not sure if this should be made a singleton; is it possible for an application to
        need open connections to more than one Shotgun instance (eg., sandbox and staging)
        at the same time?
    
    .. versionadded:: v00_03_00
    """
    
    def __init__(self, host=None, user=None, skey=None):
        """
        Setup the new shotgun connection.

        :param host:
            Shotgun host to connect to.
        :param user:
            Shotgun script user to connect as.
        :param key:
            Shotgun script key to use.       
  
        .. versionadded:: v00_03_00
        """
        self.host = host
        if not self.host and os.environ.has_key('GRENADE_SG_HOST'):
            self.host = os.environ['GRENADE_SG_HOST']
            
        self.user = user
        if not self.user and os.environ.has_key('GRENADE_SG_USER'):
            self.user = os.environ['GRENADE_SG_USER']
            
        self.skey = skey
        if not self.skey and os.environ.has_key('GRENADE_SG_SKEY'):
            self.skey = os.environ['GRENADE_SG_SKEY']

        self.session = None
        
    def connect(self, log):
        """
        Open the connection.
        
        :param log:
            A logger to record status messages to.
        :returns:
            The Shotgun session.
            
        .. todo::
            Add some more validation code (what should happen if the connection is already open for example?)
            
        .. versionadded:: v00_03_00
        """
        try:
            log.info('\nConnecting to %s as %s with key %s...' % (self.host, self.user, self.skey))
            
            self.session = Shotgun(self.host, self.user, self.skey)
            
            log.info('\tConnection established!\n')
        except Exception, e:
            message = '\tUnable to connect to shotgun server : %s\n' % str(e)
            
            log.error(message)
            raise GrenadeConnectionError(message)
            
        return self.session
        
    def get_session(self):
        """
        Get a handle on the Shotgun connection.
        
        :returns:
            The shotgun session, None if not connected.
            
        .. versionadded:: v00_03_00
        """
        return self.session

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

