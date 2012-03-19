****************
Pulling the Pin!
****************

:Release: |version|
:Date: |today|

Grenade provides a high(er) level Python interface to Shotgun, wrapping the
standard Shotgun API within an object-oriented library. It provides a more
convenient means of performing standard operations (e.g., creating new Notes)
compared to using the regular API.

Finding entities
================

The Grenade API provides a convenient means of querying Shotgun for an entity
(or entities) of interest - ``grenade.common.model.find``. This module method
allows you to search Shotgun for entities which match the provided search
criteria (specified in terms of a Shotgun API compatible list of search 
filters).   The method operates in two modes:

    * ``FIND_ONE``, which will return a single matching Grenade model, and
    * ``FIND_ALL``, the default mode of operation, which returns a list of Grenade models that match the provided search filter.

Here's an example::

    from grenade.common.model import find, FIND_ONE
    
    note = find(session, 'Note', [['id', 'is', 1]], mode=FIND_ONE)
    many = find(session, 'Note', [['subject', 'contains', '100a_010']])
    
The find method also includes support for sorting results via order by rules
defined using the same syntax as the standard Shotgun API::

	from grenade.common.model import find
	
	shots = find(session, 'Shot', [['code', 'contains', '21a']], [{'field_name': 'code', 'direction': 'desc'}])
	for shot in shots:
		print shot['code']
		
Result sorting will only apply when the ``FIND_ALL`` search mode is in effect.

Working with models
===================

In Grenade, Shotgun entities are encapsulated within models that map to a 
single Shotgun entity instance of the entity type in question. The models
support standard CRUD operations:

    * ``.create()``, create a new Shotgun entity, with field values set to the currently assigned model properties.
    * ``.read()``, refresh the model properties with the latest values for each field within the associated Shotgun entity.
    * ``.update()``, update modified model properties on the corresponding Shotgun entity.
    * ``.delete()``, delete the corresponding Shotgun entity for the model.

Grenade models also contain a set of name:value properties (accessible via 
dict-style *__getitem__*, *__setitem__*, accessors) that match up with the 
fields of the corresponding Shotgun instance.

Here's an example that demonstrates how to utilise some of these features::

    from grenade.common.model import ModelBuilder
    from grenade.utils.connection import Connection
    from rodin.logging import get_logger 

    session = Connection().connect(get_logger('grenade.example')) 

    note = ModelBuilder()(session, 'Note', None)

    note['project'] = session.find_one('Project', filters=[['sg_short_name', 'is', 'hf2']])
    note['user'] = session.find_one('HumanUser', filters=[['login', 'is', 'luke.cole']])
    note['addressings_to'] = [session.find('Group', filters=[['code', 'is', 'vfx']])]
    note['subject'] = 'grenade test note'
    note['content'] = 'boom!'
    note['sg_status_list'] = 'clsd'
    note['sg_note_type'] = 'Delivery'

    note = note.create()

    print note['id']    # id is not set until .create() is called

    note['subject'] = 'updated grenade test note'

    note.update()       # returns a list of modified note fields

Using a model translator
========================

Grenade also provides support for model translators, which may be attached to
a model in order to simplify (or customise) the format in which model property 
values need to be specified.

Model translators provide support for the Grenade CLI in particular, making it
easier for users to specify various arguments on the command line.

The following example demonstrates how to make use of a model translator (notice
in particular how it is much simpler to specify the *project*, *user* and 
*addressings_to* Note properties, when compared to the example in the section 
above)::

    from grenade.common.model import ModelBuilder
    from grenade.translators.note import NoteTranslator
    from grenade.utils.connection import Connection
    from rodin.logging import get_logger 

    session = Connection().connect(get_logger('grenade.example')) 

    note = ModelBuilder()(session, 'Note', NoteTranslator(session))

    note['project'] = 'hf2'
    note['user'] = 'luke.cole'
    note['addressings_to'] = ['vfx']
    note['subject'] = 'translated grenade test note'
    note['content'] = 'boom!'
    note['sg_status_list'] = 'clsd'
    note['sg_note_type'] = 'Delivery'

    note = note.create()

    print note['id']    # id is not set until .create() is called
