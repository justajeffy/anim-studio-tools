#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

class GrenadeError(Exception):
    """
    Base error used when things go wrong in Grenade, all other Grenade exceptions
    should derive from this.
    
    .. versionadded:: v00_01_00
    """
    
    pass

class GrenadeConnectionError(GrenadeError):
    """
    Raised when Grenade is unable to connect to Shotgun.
    
    .. versionadded:: v00_03_00
    """
    
    pass

class GrenadeValidationError(GrenadeError):
    """
    Raised when the data provided to Grenade is invalid.
    
    .. versionadded:: v00_01_00
    """
    
    pass

class GrenadeModelCreateError(GrenadeError):
    """
    Raised when there is an error creating a model.
    
    .. versionadded:: v00_02_00
    """
    
    pass

class GrenadeModelReadError(GrenadeError):
    """
    Raised when there is an error reading a model.
    
    .. versionadded:: v00_02_00
    """
    
    pass

class GrenadeModelUpdateError(GrenadeError):
    """
    Raised when there is an error updating a model.
    
    .. versionadded:: v00_02_00
    """
    
    pass

class GrenadeModelDeleteError(GrenadeError):
    """
    Raised when there is an error deleting a model.
    
    .. versionadded:: v00_02_00
    """
    
    pass

class GrenadeModelPropertyReadOnlyError(GrenadeError):
    """
    Raised when an attempt is made to edit a read-only model property.
    
    .. versionadded:: 0.9.0
    """
    
    pass

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

