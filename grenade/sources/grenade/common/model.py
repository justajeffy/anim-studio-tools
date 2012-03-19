#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..config import mappings, unsupported_api_datatypes

from ..common.error import GrenadeConnectionError, GrenadeModelCreateError, GrenadeModelReadError, GrenadeModelUpdateError, \
    GrenadeModelDeleteError, GrenadeValidationError, GrenadeModelPropertyReadOnlyError
from ..common.property import Property
from ..common.translator import Translator

FIND_ONE = 1
FIND_ALL = 2

def find(session, identifier, filters, order=[], mode=FIND_ALL, fields=None):
    """
    Retrieve model instances that match the specified Shotgun filters.
    
    :param session:
        An active Shotgun session.
    :param identifier:
        Identifier that corresponds to the Shotgun entity type of interest. Used to instantiate results.
    :param filters:
        Shotgun filters defining the entities to be found.
    :param order:
        An optional list of sorting rules, as per the standard Shotgun API find() method (e.g.,
        [{'field_name':'id','direction':'desc'}]). Only applies when used with the FIND_ALL mode as described below.
    :param mode:
        The mode in which to operate (FIND_ONE or FIND_ALL)
    :param fields:
        A list of fields to query the schema for - optimisation over returning all fields for the 
        entity which can be slow and overkill in some situations.
    :returns:
        Find results in the form of (a) model instance(s) based on the selected mode:
        
            FIND_ONE: A single result matching the first Shotgun entity which was found.
            FIND_ALL: A list of results matching all the Shotgun entities which were found.
            
        The results are ordered by the specified sorting rules, if present.
            
    .. versionadded:: v00_02_00
    .. versionchanged:: v00_03_00
        Updated to expect a model identifier, rather than the actual model class to be returned. Use the
        model builder to construct results based on the Shotgun schema of the entity being found.
    .. versionchanged:: v00_05_01
        Updated to pre-load the model builder with the schema for the model to build.
    .. versionchanged:: v00_08_00
        Added support for sorting of results.
    .. versionchanged:: v00_08_03
        In FIND_ONE mode, throw an exception when no entities for the specified identifier and filters 
        were found.
    .. versionchanged:: 0.10.0
        Added the optional ``fields`` keyword argument.
    .. versionchanged:: 1.0.0
        Filter out 'summary' and 'pivot_column' fields specified within the schema (if no specific fields
        have been requested by the user), as these aren't presently viewable via the Shotgun API.
    .. versionchanged:: 1.1.0
        Unsupported datatypes definition has been moved into the grenade configuration.
    .. versionchanged:: 1.4.0
    	Wrap shotgun API find() calls in try except statements, to capture any errors and return 
    	developer/user-friendly feedback.
        
    .. todo::
        If a list of fields to retrieve has been requested by the user, check each one to ensure that it
        is "viewable" via the Shotgun API (ie, not of type 'summary' or 'pivot_column'), if an unsupported
        type is requested, throw an exception.
    """
    # step 1: we need to know the schema of the entity we are querying
    schema = {}
    
    if fields is None:
        schema = session.schema_field_read(identifier)
        for key in schema.keys():
            if schema[key]['data_type']['value'] in unsupported_api_datatypes:
                del schema[key]
    else:
        # assumption: the user didn't request any fields that aren't viewable via the shotgun API ...
        for field in fields:
            schema.update(session.schema_field_read(identifier, field))
    
    builder = ModelBuilder(schema)
    
    # step 2: now we can see if there are any results to find
    if mode == FIND_ALL:
        try:
            entities = session.find(identifier, filters, schema.keys(), order)
        except ValueError, e:
            raise GrenadeConnectionError('An internal shotgun error was encountered on session.find("%s")' % identifier)
    
        results = []
        for entity in entities:
            results.append(builder(session, identifier, None, **entity))
                
        return results
    elif mode == FIND_ONE:
        try:
            entity = session.find_one(identifier, filters, schema.keys())
        except ValueError, e:
            raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("%s")' % identifier)
        
        if not entity:
            raise GrenadeModelCreateError('No %s entities found using filters : %s' % (identifier, filters))
        else:
            return builder(session, identifier, None, **entity)
    else:
        raise NotImplementedError

class ModelBuilder(object):
    """
    Model factory that may be used to generate Grenade models that represent Shotgun entities.
    
    .. versionadded:: v00_03_00
    .. versionchanged:: v00_05_01
        Allow pre-loading of a schema to use when instantiating model instances - greatly
        increases performance of find() calls (or multiple instantiation operations).
    """
    
    def __init__(self, schema=None):
        """
        Setup the new factory instance.
        
        :param schema:
            Schema to use when instantiating model instances.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: v00_05_01
            Added support for optional schema argument.
        """
        self.schema = schema
    
    def __call__(self, *args, **kwargs):
        """
        Factory method; based on arguments, return the specified Model object.
        
        :param session:
            An active Shotgun session
        :param identifier:
            Identifier for the type of Model to create.
        :param translator:
            Optional property translator.
        :param kwargs:
            Model specific initialisation parameters.
        
        .. versionadded:: v00_03_00
        .. versionchanged:: v00_05_01
            Only lookup the schema for the specified model, if it has not already been pre-loaded.
        .. versionchanged:: 0.9.0
            Force property value assignment, even if the property is read-only, during model construction.
        .. versionchanged:: 1.0.0
            Improve robustness of property set checks.
        .. versionchanged:: 1.0.0
            Filter out 'summary' and 'pivot_column' fields specified within the schema, as these aren't 
            presently viewable via the Shotgun API.
        .. versionchanged:: 1.1.0
            Use unsupported api datatypes list imported from grenade configuration.
        """
        session, identifier, translator = args

        if session:
            try:
                if not self.schema:
                    self.schema = session.schema_field_read(identifier)
                    for key in self.schema.keys():
                        if self.schema[key]['data_type']['value'] in unsupported_api_datatypes:
                            del self.schema[key]
                    
                object = Model(session, identifier, translator, self.schema)
                
                if kwargs:
                    for key in kwargs.keys():
                        if object.properties.has_key(key):
                            object.set_property(key, kwargs[key], force=True)
        
                    object._purify()
                
                return object
                
            except Exception, e:
                raise GrenadeValidationError("Invalid model definition! (%s)" % e)
        else:
            raise GrenadeConnectionError("Invalid Shotgun session!")

class Model(object):
    """
    Superclass for grenade models that represent Shotgun entities.
    
    .. versionadded:: v00_01_00
    .. versionchanged:: v00_02_00
        Added translator support, auto-property creation via schema, dict-style accessors, value purification.
        Explicitly inherit from object.
    .. versionchanged:: v00_03_00
        Added model identifier and schema definition, updated CRUD methods to pass in the identifier rather than
        the class name, in order to support dynamic model building.
        
    .. todo::
        If a translator is present, the create and update methods should probably make a pass over all the 
        properties prior to writing them out to ensure that any values with the check field set have been converted
        into the correct format ... ?
    """
    
    def __init__(self, session, identifier=None, translator=None, schema={'id':None}):
        """
        Setup the new model instance.
        
        :param sesson:
            An active Shotgun session.
        :param identifier:
            An identifier for the model (usually indicates type).
        :param translator:
            An optional translator to convert in-bound property values.
        :param schema:
            Dictionary which represents the Shotgun schema of this model.
            
        .. versionadded:: v00_01_00
        .. versionchanged:: v00_02_00
            Accept an optional translator and schema definition (used to auto create properties).
        .. versionchanged:: v00_03_00
            Updated to accept a model identifier, change the schema argument to a dictionary, store the schema internally, set the
            data type and valid values on each property. Include a default translator.
        .. versionchanged:: v00_05_02
            Fixed an incorrect key name in the schema dictionary.
        .. versionchanged:: v00_08_00
            Set the default value specified for each schema field in shotgun on the corresponding model property.
        .. versionchanged:: 0.9.0
            Set the read-only flag on model properties where the corresponding schema field in shotgun is not editable.
        """
        
        self.session = session
        self.identifier = identifier
        self.translator = translator
        self.properties = {}
        
        # create a default translator, if none has been specified
        if not self.translator:
            self.translator = Translator(self.session)
        
        # define the schema, ensure that an id entry is present
        self.__schema__ = schema.keys()
        
        if 'id' not in self.__schema__:
            self.__schema__.append('id')
        
        # define the model properties, based on the schema, set default, data type and valid values where relevant
        for key in self.__schema__:
            if schema.has_key(key) and schema[key]:
                if schema[key]['properties'].has_key('default_value'):
                    default = schema[key]['properties']['default_value']['value']
                else:
                    default = None
                    
                if schema[key]['properties'].has_key('valid_values'):
                    values = schema[key]['properties']['valid_values']['value']
                else:
                    values = []
                
                if schema[key].has_key('editable'):    
                    readonly = not schema[key]['editable']['value']
                else:
                    readonly = False
                
                self.__add_property(key, default, data_type=schema[key]['data_type']['value'], valid_values=values, read_only=readonly)
            else:
                self.__add_property(key)
        
    def get_property(self, key):
        """
        Get the current value of the model property which matches the supplied key.
    
        :param key:
            Identifier for the property to query.
        :returns:
            The value of the specified model property.    

        .. versionadded:: v00_01_00
        """
        return self.properties[key].value
    
    def set_property(self, key, value, force=False):
        """
        Set the value of the model property which matches the supplied key.
        
        :param key:
            Identifier for the property to set.
        :param value:
            Value to assign to the specified property. This value may be translated if a translator has been
            attached to the model.
        :param force:
            Force value assignment, even if the property is read-only. This is necessary to set the field to
            an initial value when building the model - this argument should not be needed for normal use.

        .. versionadded:: v00_01_00
        .. versionchanged:: v00_02_00
            Translate the value, if a translator has been assigned to the model.
        .. versionchanged:: v00_03_00
            Always translate the value, this is safe, as the default model translator will do nothing.
        .. versionchanged:: 0.9.0
            Throw an exception if an attempt is made to set the value of a read-only model property.
        .. versionchanged:: 0.9.0
            Added an argument to permit forcing assignment of read-only model properties (needed during
            model initialisation).
        .. versionchanged:: 0.9.1
            Allow changes to a read-only field during model creation (ie, model id == None), as some 
            read-only fields in Shotgun support this.
            
        .. todo::
            Find out how to tell if a particular non-editable field permits editing at creation time, and
            implement proper support for this.
        """
        if self.properties['id'].value != None and self.properties[key].is_read_only and not force:
            raise GrenadeModelPropertyReadOnlyError('Trying to set a read-only model property : %s' % key)
        else:
            self.properties[key].set_value(self.translator.translate(key, value))
            
    def create(self):
        """
        Create an instance of this model within Shotgun.

        :returns:
            A copy of self, updated to include the id of the matching Shotgun entity which was created.

        .. versionadded:: v00_01_00
        .. versionchanged:: v00_02_00
            Raise a GrenadeModelCreateError exception when attempting to create a model which already
            has an identifier, purify property values on create.
        .. versionchanged:: v00_03_00
            Pass the model identifier into shotgun, rather than the class name.
        .. versionchanged:: 0.9.0
            Do not include read-only properties in data passed to shotgun during creation.
        .. versionchanged:: 0.9.1
            Actually, *do* include read-only properties in data passed to shotgun during creation, as
            apparently, some of these are updatable (but only at creation time).
        .. versionchanged:: 1.1.0
            There are some read-only properties that are not updatable, ever.   There is no way to
            tell the difference between these properties, and those that are updatable (but only at
            creation time) via the shotgun schema.   We have a hardcoded list of these within our
            configuration settings therefore that we use to check properties against at creation time.
       
        .. note::
            On create, this method invokes .read(), in order to populate read-only fields with the 
            correct values from Shotgun, etc.
        """
        data = {}
        
        if self.properties['id'].value != None:
            raise GrenadeModelCreateError('Trying to create a model which already has an id : %s' % self.properties['id'].value)
        
        for key in self.properties.keys():
            if key != 'id' and self.properties[key].value != None and not (mappings.has_key(self.identifier) and mappings[self.identifier].has_key('never_editable') and key in mappings[self.identifier]['never_editable']):
            	data[key] = self.properties[key].value
        
        results = self.session.create(self.identifier, data)
        self.set_property('id', results['id'], force=True)
        
        return self.read()
    
    def read(self):
        """
        Read in the model instance properties from the corresponding Shotgun entity.

        :returns:
            A copy of self, updated to include the latest property values for the matching Shotgun entity.

        .. versionadded:: v00_01_00
        .. versionchanged:: v00_02_00
            Move find style behaviour into module model find method, simply refresh current model properties
            with matching fields in Shotgun instead.
        .. versionchanged:: v00_03_00
            Pass the model identifier into shotgun, rather than the class name, also, when looping through the results, use our
            property keys as the index into the results, as the results keys may contain properties that the model is not yet
            aware of. Directly set the property values, as we don't want to translate when reading in from Shotgun.
        .. versionchanged:: 1.4.0
    		Wrap shotgun API find() calls in try except statements, to capture any errors and return 
    		developer/user-friendly feedback.
            
        .. todo::
            Implement support for new Shotgun fields that aren't in this model's schema (?)
            
        .. todo::
            Implement support for changes in the model's schema, since last time this model was read (?)
        """
        results = {}
        
        if self.properties['id'].value != None:
            try:
                results = self.session.find_one(self.identifier, [['id', 'is', self.properties['id'].value]], self.properties.keys())
            except ValueError, e:
                raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("%s")' % self.identifier)
        else:
            raise GrenadeModelReadError('Trying to read an unidentified model!')
            
        if results:
            for key in self.properties.keys():
                self.properties[key].set_value(results[key])    # we don't need to translate when reading in from shotgun
                    
            self._purify()
        else:
            self.__nullify()    # no results were found
                
        return self
    
    def update(self):
        """
        Update modified model instance properties on the corresponding Shotgun entity.
        
        :returns:
            A list of keys corresponding to the model properties which were updated in Shotgun.
        
        .. versionadded:: v00_01_00
        .. versionchanged:: v00_02_00
            Raise a GrenadeModelCreateError if trying to update an unidentified model, purify properties after performing an
            update.
        .. versionchanged:: v00_03_00
            Pass the model identifier into shotgun, rather than the class name.
        .. versionchanged:: v00_07_00
            Use the new property class accessors for the is_dirty field.
        .. versionchanged:: 0.9.0
            Do not try to update read-only fields.
        """
        results = {}
        changes = {}
        
        if self.properties['id'].value == None:
            raise GrenadeModelUpdateError('Trying to update an unidentified model!')
        
        for key in self.properties.keys():
            if key != 'id' and not self.properties[key].is_read_only and self.properties[key].get_is_dirty():
                changes[key] = self.properties[key].value
        
        if self.properties['id'].value and len(changes.keys()) > 0:
            results = self.session.update(self.identifier, self.properties['id'].value, changes)
            self._purify()
        
        return changes.keys()
    
    def delete(self):
        """
        Delete the corresponding Shotgun entity for this model instance.

        :returns:
            A boolean indicating the success of the operation.   If successful, the model instance is invalidated.

        .. versionadded:: v00_01_00
        .. versionchanged:: v00_02_00
            Raise a GrenadeModelDeleteError if trying to update an unidentified model.
        .. versionchanged:: v00_03_00
            Pass the model identifier into shotgun, rather than the class name.
        """
        if self.properties['id'].value:
            results = self.session.delete(self.identifier, self.properties['id'].value)
            if results:
                self.__nullify()
                return True
        else:
            raise GrenadeModelDeleteError('Trying to delete an unidentified model!')
        
        return False
    
    def _purify(self):
        """
        Mark all modified model properties as clean.
        
        .. versionadded:: v00_02_00
        .. versionchanged:: v00_07_00
            Use the new property class accessors for the is_dirty field.
        .. versionchanged:: v00_07_00
            Also reset the check value, if present, during purification.
        """
        for key in self.properties.keys():
            self.properties[key].set_is_dirty(False)
            self.properties[key].reset_check()
    
    def __nullify(self):
        """
        Invalidate the model instance on deletion.
        
        .. versionadded:: v00_01_00
        """
        for key in self.properties.keys():
            self.properties[key] = Property(None)

    def __add_property(self, key, value=None, data_type=None, valid_values=[], read_only=False):
        """
        Add a new property to the model with the supplied key.
        
        :param key:
            Identifier for the property to add.
        :param value:
            Optional intitial value to assign to the property.
        :param data_type:
            Optional data type to assign to the property.
        :param valid_values:
            Optional valid values list for this property.
        :param read_only:
            Optional flag to indicate if this property is read-only.

        .. versionadded:: v00_03_00
        .. versionchanged:: v00_08_00
            Added a new argument to specify the initial value to assign to the field.
        .. versionchanged:: 0.9.0
            Added support for read-only properties.
        """
        if not self.properties.has_key(key):
            self.properties[key] = Property(value, data_type, valid_values, read_only)
        else:
            raise GrenadeValidationError('Property %s already exists!' % key)
    
    def __getitem__(self, key):
        """
        self[key] style container access.

        :param key:
            Identifier for the property to query.
        :returns:
            The value of the specified model property.

        .. versionadded:: v00_02_00
        """
        return self.get_property(key)
    
    def __setitem__(self, key, value):
        """
        self[key] = value style container access.

        :param key:
            Identifier for the property to set.
        :param value:
            Value to assign to the specified property. This value may be translated if a translator has been
            attached to the model.
        
        .. versionadded:: v00_02_00
        """
        self.set_property(key, value)
            
    def __repr__(self):
        """
        Output a generic representation of the model instance.

        :returns:
            Printable representation of self.

        .. versionadded:: v00_01_00
        """   
        return '%s' % self.properties

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

