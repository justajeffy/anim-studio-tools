##
#   \namespace  reviewTool.api.io
#
#   \remarks    Defines different input/output systems for saving and loading review tool data
#   
#   \author     eric.hulser@drdstudios.com
#   \author     Dr. D Studios
#   \date       08/18/11
#

# define global variables
_loaded     = False

# define global functions
def init():
    global _loaded
    if ( not _loaded ):
        _loaded = True
        
        import os.path, glob
        filenames = glob.glob( os.path.split( __file__ )[0] + '/*.py' )
        for filename in filenames:
            modname = os.path.basename( filename ).split( '.' )[0]
            
            # do not import the init module
            if ( modname != '__init__' ):
                package = '%s.%s' % ( __name__, modname )
                __import__( package )
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

