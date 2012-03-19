#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from copy import deepcopy
from operator import itemgetter
from random import randrange

class MockShotgun(object):
    """
    Mock Shotgun session for use within unit tests. Implements a subset of the Shotgun V3 API,
    runs against a local "cache" of test data.
    
    >>> def setup(self):
    >>>     self.session = MockShotgun( ... )
    >>>
    >>> def test_shotgun(self):
    >>>     result = use_shotgun( ..., self.session)
    
    .. versionadded:: 0.2.0
    """
    def __init__(self, host='', user='', skey='', schema=[], data=[], first_id=1):
        """
        Setup the mock Shotgun instance.
        
        :param host:
            Shotgun host URL to connect to.
        :param user:
            Shotgun user to connect as.
        :param skey:
            Shotgun script key to connect with.
        :param schema:
            List of schema definition dictionaries.
        :param data:
            List of entity data dictionaries.
        
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.3.0
        	Define the base_url session property (apparently it is the same as the host URL)
        .. versionchanged:: 0.8.0
            Added an extra keyword argument to permit setting the id generator's start value.
        """
        self.host = host
        self.user = user
        self.skey = skey
        
        self.base_url = self.host
        
        # make sure we have our own internal copy of these to shield ourselves from issues caused by external manipulations:
        self.schema = deepcopy(schema)
        self.data = deepcopy(data)
        
        self.__operands = {'is':'==', 'is_not':'!='}
        self.__id_sequence = self.__generate_id(first_id)
    
    def find(self, entity_type, filters, fields=None, order=[]):
        """
        Find entities that match the search criteria.
        
        :param entity_type:
            The type of the entity to look for.
        :param filters:
            Search filters to apply (currently only supports 'is' and 'is not' filters).
        :param fields:
            Entity fields to return.
        :param order:
            Sorting rules to apply to the results.
        :returns:
            List of Shotgun API compatible data dictionaries for the selected result, None if no result found.
            
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.13.2
            Fix buggy find behaviour; create_search assembles multiple filters into a single test, don't loop
            over them, otherwise we execute the test multiple times, and return duplicated results.
        """
        results = []
        for entity in self.data:
            if entity['type'] == entity_type:
                if eval(self.__create_search(filters)):
                    result = {}
                    result['type'] = entity_type        # ensure the type is included
                    
                    if fields:
                        for field in fields:
                            result[field] = entity[field]
                        
                    if 'id' not in result.keys():
                        result['id'] = entity['id']     # ensure the entity id is included
                    
                    results.append(result)
        
        order.reverse() # rules must be reversed to obtain results the user expects (relies on python stable sorting behavior)
        for rule in order:
            results = sorted(results, key=itemgetter(rule['field_name']), reverse = True if rule['direction'] == 'desc' else False)
        
        return results
    
    def find_one(self, entity_type, filters, fields=None):
        """
        Find the first entity that matches the search criteria.
        
        :param entity_type:
            The type of the entity to look for.
        :param filters:
            Search filters to apply (currently only supports a 'is' or 'is not' filters).
        :param fields:
            Entity fields to return.
        :returns:
            Shotgun API compatible data dictionary for the selected result, None if no result found.
            
        .. versionadded:: 0.2.0
       	.. versionchanged:: 0.3.0
       		Don't try to set fields on the result that we don't have data for.
       	.. versionchanged:: 0.13.2
            Fix buggy find behaviour; create_search assembles multiple filters into a single test, don't loop
            over them, otherwise we execute the test multiple times, and return duplicated results.
        """
        result = {}
        result['type'] = entity_type    # ensure the type is included
        for entity in self.data:
            if entity['type'] == entity_type:
                if eval(self.__create_search(filters)):
                    if fields:
                        for field in fields:
                            if field in entity:
                                result[field] = entity[field]
                        
                    if 'id' not in result.keys():
                        result['id'] = entity['id']     # ensure the entity id is included
                        
                    return result
            
        return None     # no match was found
    
    def create(self, entity_type, data, return_fields=None):
        """
        Create a new entity.
        
        :param entity_type:
            The type of the entity to create.
        :param data:
            Properties dictionary for the entity being created.
        :param return_fields:
        	Optional list of fields to return (otherwise, only id and type are returned).
        :returns:
            Shotgun API compatible entity dictionary for the new instance.
            
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.3.0
            Use a sequential id generator to populate the id field, rather
            than a random value between 1 and 10.
        .. versionchanged:: 0.3.0
            Implement support for return fields keyword argument.
        .. versionchanged:: 0.3.0
            Ensure sg_status_list field is present on shot entities, even
            if the inbound data doesn't define it ...
        .. versionchanged:: 0.8.0
            Implement code to fill un-populated data fields with default values
            if they are defined by the entity schema.
            
        .. todo::
            Remove hacky shot sg_status_list field set operation ...
        """
        data['id'] = self.__id_sequence.next()
        data['type'] = entity_type
        
        # kinda hacky, but the simplest way of achieving this for now ...
        if entity_type in ['Shot'] and not data.has_key('sg_status_list'):
            data['sg_status_list'] = ''
        
        # check the schema for any fields with default values that have not been populated yet
        schema = None
        for entry in self.schema:
            if entry.has_key(entity_type):
                schema = entry[entity_type]
                break
        
        if schema:
            for field in schema.keys():
                # make sure the schema contains the required parts
                if schema[field].has_key('properties') and schema[field]['properties'].has_key('default_value') and schema[field]['properties']['default_value'].has_key('value'):
                    # now perform the actual test - is there a field defined in the schema 
                    # with a default value assigned that hasn't been populated already by the supplied data?
                    if field not in data.keys() and schema[field]['properties']['default_value']['value']:
                        data[field] = schema[field]['properties']['default_value']['value']
        
        self.data.append(data)
        
        result = {'id': data['id'], 'type': data['type']}
        if return_fields:
            for field in return_fields:
                if field in data and field not in ['id', 'type']:
                    result[field] = data[field]
        
        return result
    
    def update(self, entity_type, entity_id, data):
        """
        Update an existing entity.
        
        :param entity_type:
            The type of the entity being updated.
        :param entity_id:
            Id of the entity to apply updates to.
        :param data:
            Property dictionary containing fields to be updated on the entity.
        :returns:
            Shotgun API compatible entity dictionary for the updated instance.
            
        .. versionadded:: 0.2.0
        """
        for entity in self.data:
            if entity['id'] == entity_id and entity['type'] == entity_type:
                for key in data.keys():
                    entity[key] = data[key]
                return entity 
            
        return None     # no matching entity was found
    
    def delete(self, entity_type, entity_id):
        """
        Delete an entity.
        
        :param entity_type:
            Type of entity to delete.
        :param entity_id:
            Id of the entity to be deleted.
        :returns:
            True on success, False on failure.
            
        .. versionadded:: 0.2.0
        """
        for entity in self.data:
            if entity['id'] == entity_id and entity['type'] == entity_type:
                self.data.remove(entity)
                return True
            
        return False    # no matching entity was found
    
    def schema_field_read(self, entity_type, field_name=None):
        """
        Read the schema definition for a specified entity/field.
        
        :param entity_type:
            Type of entity to read the schema of.
        :param field_name:
            Schema field to read.
        :returns:
            Selected schema definition, None if no match was found.
            
        .. versionadded:: 0.2.0
        .. versionchanged:: 0.6.2
        """
        for entry in self.schema:
            if not field_name:
                if entry.has_key(entity_type):
                    return entry[entity_type]
            else:
                for key in entry.keys():
                    if key == entity_type and field_name in entry[key].keys():
                        return {field_name:entry[key][field_name]}
        
        return None     # no matching schema was found

    def upload_thumbnail(self, entity_type, entity_id, thumbnail_path):
    	"""
    	Upload a thumbnail for the specified entity.
    	
    	:param entity_type:
    	    Type of entity to upload the thumbnail to.
    	:param entity_id:
    	    Id of the specific entity to attach the thumbnail to.
        :param thumbnail_path:
            Path to the thumbnail image to upload.
        
        .. versionadded:: 0.3.0
            
        .. note::
            This is an empty skeleton/wrapper method - it doesn't actually do anything.
    	"""
        pass
    
    def __generate_id(self, start):
        """
        Internal helper method to generate sequential entity ids.
        
        :param start:
            The initial value from which to start the id sequence.
        :returns:
            The next id number in the sequence.

        .. versionadded:: 0.3.0
        """
        next = start
        while True:
            yield next
            next += 1
    
    def __create_search(self, filters):
        """
        Internal helper method to assemble a search code fragment.
        
        :param filters:
            List of Shotgun API compatible search filters.
        :returns:
            An un-evaluated search code fragment.
            
        .. versionadded:: 0.2.0
        """
        result = ''
        
        for filter in filters:
            if isinstance(filter[2], str):              # matching a string
                template = "entity['%s'] %s '%s'"
            elif isinstance(filter[2], dict):           # matching a entity
                template = "entity['%s']['%s'] %s %s"
            else:
                template = "entity['%s'] %s %s"         # matching an id
            
            if filter[1] not in self.__operands:
                raise NotImplementedError('Operator %s not supported' % filter[1])
            else:
                if isinstance(filter[2], dict):
                    test = template % (filter[0], 'id', self.__operands[filter[1]], filter[2]['id'])
                else:
                    test = template % (filter[0], self.__operands[filter[1]], filter[2])
                result += test
                
                if filter != filters[-1]:
                    result += ' and '
        
        return result

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

