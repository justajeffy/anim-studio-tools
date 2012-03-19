# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

class MockLog(object):
    """
    Mock rodin style logger, to aid in unit testing.
    
    >>> def setup(self):
    >>>     self.log = MockLog()
    >>>
    >>> def test_log(self):
    >>>     do_something( ..., self.log)
    >>>
    >>>     assert len(self.log.messages['error']) == 0    
    >>>     assert len(self.log.messages['debug']) == 1
    >>>     
    >>>     assert self.log.messages['debug'][0] == 'Something happened.'
    
    .. versionadded:: 0.2.0
    """
    def __init__(self):
        self.messages = {'info':[], 'error':[], 'exception':[], 'debug':[], 'warn':[]}
        
    def info(self, message):
        self.messages['info'].append(message)
        
    def error(self, message):
        self.messages['error'].append(message)
        
    def exception(self, message):
        self.messages['exception'].append(message)
        
    def debug(self, message):
        self.messages['debug'].append(message)
        
    def warn(self, message):
        self.messages['warn'].append(message)
        
    def clear(self):
        self.messages['info'] = []
        self.messages['error'] = []
        self.messages['exception'] = []
        self.messages['debug'] = []
        self.messages['warn'] = []

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

