# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

class MockLabels(object):
    """
    Mock labels object, to aid in unit testing.
    
    Emulates Tank behavior where labels are also properties!
    
    .. versionadded:: 0.2.0
    """
    def __init__(self, labels={}):
        self.labels = labels
    
    @property
    def RoleType(self):
        return self.labels['RoleType']

class MockSystemObject(object):
    """
    Mock tank revision system object, to aid in unit testing.
    
    .. note::
        Presently only supports the system name and type name.
    
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.15.1
        Added support for tkid property.
    """
    def __init__(self, tkid="@ABC_123", name="test_name", type_name="test_type"):
        self.tkid = tkid
        self.name = name
        self.type_name = type_name
        
class MockAssetObject(object):
    """
    Mock tank revision asset object, to aid in unit testing.
    
    .. note::
        Presently only supports the asset container, system object and labels.
    
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.12.0
        Added support for asset container properties.
    """
    def __init__(self, container="test_asset", labels={}):
        self.container = container  # supports strings or objects for simplified testing ...
        self.system = MockSystemObject()
        self.labels = MockLabels(labels)
        self.properties = MockProperties()
    
    def save(self):
        """
        Pretend to save changes to the asset to Tank.
        
        .. versionadded:: 0.12.0
        """
        return True
        
    def __repr__(self):
        if type(self.container) == str:
            return self.container
        else:
            return str(self.container)
        
class MockProperties(object):
    """
    Mock tank revision properties, to aid in unit testing.
    
    >>> revision = MockRevision({'system.type_name':'RevisionA', 'properties':{'pipeline_data':None}})
    >>>
    >>> assert revision.properties.pipeline_data == None
    
    .. note::
        Presently only supports the "contents" and "pipeline_data" properties.
    
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.7.0
        Added support for the "is_locked" property (currently only available on department roles)
    """
    is_locked = None
    
    def __init__(self, properties={}):
        self.properties = properties
        
        if self.properties.has_key('is_locked'):
            self.is_locked = self.properties['is_locked']
        
    @property
    def contents(self):
        return self.properties['contents']
    
    @property
    def pipeline_data(self):
        return self.properties['pipeline_data']
    
    def keys(self):
        return self.properties.keys()

class MockRevision(object):
    """
    Mock tank revision, to aid in unit testing.   This mock object implements (often
    in quite a hacky way) a very limited subset of the behaviour supported by regular 
    tank revision objects.

    >>> revision = MockRevision({'system.type_name':'RevisionA', 'asset.system.type_name':'ContainerA_v1'})
    >>> 
    >>> assert revision.system.type_name == 'RevisionA'            # check the type of the revision
    >>> assert revision.asset.system.type_name == 'ContainerA_v1'  # check the type of the revision's container
    
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.7.0
        Added an empty save() wrapper method.
    .. versionchanged:: 0.12.0
        Added support for asset properties.
    .. versionchanged:: 0.15.1
        Added support for tkid property.
    
    .. todo::
        Ultimately, the need for this mock object may be removed via the implementation
        of an "automated" tank scratch for unit testing environment.   In such a case,
        actual tank revision objects against a controlled set of test data may be used
        instead.
        
    .. todo::
        Is the behaviour of __repr__ actually correct (with regard to equivalent tank functionality) ...?
    """
    def __init__(self, data):
        self.asset = MockAssetObject()
        self.system = MockSystemObject()
        
        if data.has_key('asset'):
            self.asset.container = data['asset']
        
        if data.has_key('asset.labels'):
            self.asset.labels = MockLabels(data['asset.labels'])
            
        if data.has_key('asset.properties'):
            self.asset.properties = MockProperties(data['asset.properties'])
            
        if data.has_key('asset.system.type_name'):
            self.asset.system.type_name = data['asset.system.type_name']
        
        if data.has_key('system.tkid'):
            self.system.tkid = data['system.tkid']
        
        if data.has_key('system.name'):
            self.system.name = data['system.name']
        
        if data.has_key('system.type_name'):
            self.system.type_name = data['system.type_name']
            
        if data.has_key('properties'):
            self.properties = MockProperties(data['properties'])
    
    def save(self):
        """
        Pretend to save changes to the revision to Tank.
        
        .. versionadded:: 0.7.0
        """
        return True
        
    def __repr__(self):
        """
        Return a representation of this revision.
        
        .. versionchanged:: 0.15.1
            Updated implementation to be consistent with behaviour in Tank.
        """
        return "%s(%s, %s)" % (self.system.type_name, self.system.name, self.asset)

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

