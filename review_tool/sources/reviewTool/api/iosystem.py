##
#   \namespace  reviewTool.api.io
#
#   \remarks    Defines different input/output exporters for saving
#               and loading data from files
#   
#   \author     Dr. D Studios
#   \date       08/18/11
#

from . import io

class IOSystem(object):
    _systems = {}
    
    def __init__( self, systemName, fileType = '', imports = False, exports = False ):
        self._systemName    = systemName
        self._fileType      = fileType
        self._imports       = imports
        self._exports       = exports
    
    def exports( self ):
        return self._exports
    
    def fileType( self ):
        return self._fileType
    
    def imports( self ):
        return self._imports
    
    def load( self, filename ):
        """
                Restores the playlist context from the inputed filename
                
                :param      filename:
                :type       <str>:
                
                :return     <PlaylistContext> || None:
        """
        return None
    
    def save( self, filename, clips ):
        """
                Saves the inputed clips to a file
                
                :param      filename:
                :type       <str>:
                
                :param      clips:
                :type       <list> [ <Clip>, .. ]:
                
                :return     <bool>: success
        """
        return False
    
    def systemName( self ):
        return self._systemName
    
    @staticmethod
    def find( systemName ):
        """
                Looksup the given system by name
                
                :param  systemName:
                :type   <str>:
                
                :return     <IOSystem> || None:
        """
        io.init()
        return IOSystem._systems.get(str(systemName))
    
    @staticmethod
    def findByType( fileType ):
        """
                Looksup the given system by extension
                
                :param  fileType:
                :type   <str>:
                
                :return     <IOSystem> || None:
        """
        io.init()
        for system in IOSystem._systems.values():
            if ( system.fileType() == fileType ):
                return system
        return None
    
    @staticmethod
    def register( systemName, cls, fileType = '', imports = False, exports = False ):
        """
                Regsiters the inputd IOSystem class as an input/output
                mechanism for loading and saving review tool information
                
                :param  fileType:
                :type   <str>:
                
                :param  cls:
                :type   <subclass of IOSystem>:
                
                :param  imports:
                :type   <bool>: 
        """
        IOSystem._systems[str(systemName)] = cls(systemName,fileType,imports,exports)
    
    @staticmethod
    def systems():
        io.init()
        return IOSystem._systems.values()
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

