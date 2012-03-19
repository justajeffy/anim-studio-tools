# 
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
# 

class MockDatabaseError(Exception):
    """
    Mock SqlAlchemy database error.
    
    Thrown by mock database objects when various different error conditions are 
    encountered (e.g., when support for some feature hasn't been implemented).
    
    .. versionadded:: 0.2.0
    """
    pass

class MockResultProxy(object):
    """
    Mock SqlAlchemy database result proxy.
    
    >>> self.database = MockDatabase()
    >>>
    >>> result = self.database.query( ... ).filter( ... ).one()
    
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.3.0
        Return None if empty on call to :meth:`first`.
    .. versionchanged:: 0.3.0
        Added :meth:`options` method.
    """
    def __init__(self, data=[]):
        self.data = data
    
    def first(self):
        if self.data:
            return self.data[0]
        else:
            return None
    
    def options(self, *args):
        return self
    
    def one(self):
        return self.first()
    
    def all(self):
        return self.data

class MockDatabaseFilter(object):
    """
    Mock SqlAlchemy database filter parser.
    
    >>> self.database = MockDatabase()
    >>>
    >>> results = self.database.query( ... ).filter( ... )
    
    .. versionadded:: 0.2.0
    .. versionchanged:: 0.14.4
    
    .. todo::
        Implement proper filter behaviour ... if we find we need it.
    """
    def __init__(self, data=[]):
        self.data = data

    def filter(self, filter):
        return MockResultProxy(self.data)
    
    def all(self):
        # pass through to the result proxy in cases where the results aren't filtered at all
        return MockResultProxy(self.data).all()
    
class MockDatabase(object):
    """
    Mock SqlAlchemy database, to aid in unit testing.
    
    >>> def setup(self):
    >>>     self.database = MockDatabase()
    >>>
    >>> def test_database(self):
    >>>     thing_data = { ... }
    >>>     
    >>>     result = create_thing(thing_data, self.database)
    >>>     
    >>>     assert result.id >= 0
    >>>     assert result in self.database.data
    >>>     assert len(self.database.data) == 1
    
    .. versionadded:: 0.2.0
    
    .. todo::
        Implement proper detection of non-unique record inserts - currently, all that
        happens is we check to see if a record has already been written, if so, subsequent
        writes thrown an error ... lame!
    """
    def __init__(self):
        self.pending = []
        self.data = []
        self.unique = False     # switch on to prevent adding non-unique records
        
        def key_gen():
            count = 0
            while True:
                yield count
                count += 1
                
        self.keys = key_gen()
    
    def add(self, record):
        self.pending.append(record)
    
    def commit(self):
        if self.pending:
            if self.unique and self.data:
                raise MockDatabaseError("Unique mode enabled!")
            
            for record in self.pending:
                if not record.id:
                    record.id = self.keys.next()
                else:
                    for comitted in self.data:
                        if comitted.id == record.id:
                            self.data.remove(comitted)

                self.data.append(record)
            
            self.pending = []
            
    def query(self, model):
        filter_set = []
        for record in self.data:
            if record.__class__ == model:
                filter_set.append(record)
                
        return MockDatabaseFilter(filter_set)
    
    def rollback(self):
        self.pending = []

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

