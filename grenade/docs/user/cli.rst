**********************
Command Line Interface
**********************

:Release: |version|
:Date: |today|

Grenade provides a basic command line interface to Shotgun, which may be used
to perform CRUD operations on Shotgun entities via the shell.

In general, Grenade command line usage takes the form::

    grenade_cl --mode <mode> --entity <entity> --args <args> [--verbose 0|1]

Where:

    * ``<mode`` specifies the operation to perform (e.g., *create*)
    * ``<entity>`` is the type of Shotgun entity to operate on (e.g., *Note*)
    * ``<args>`` are the command arguments (these depend on the selected mode)

The CLI operates in four different modes, each of which are outlined in more 
detail below.

Create 
======

Create a new instance of the specified entity within Shotgun::

    grenade_cl --mode create --entity <entity> --args <args>

Where:
   
    * ``<entity>`` is the type of Shotgun entity to operate on (e.g., *Note*)
    * ``<args>`` is a dictionary of name:value pairs describing the entity fields to set, and their assigned values. 

Example(s)::

    $ grenade_cl --mode create --entity Note --args "{'project':'hf2', 'user':'luke.cole', 'addressings_to':['luke.cole'], 'subject':'grenade test', 'content':'test', 'note_links':[{'Scene':[['id', 'is', 1]]}, {'Shot':[['id', 'is', 6174]]}], 'sg_status_list':'clsd', 'sg_note_type':'Delivery'}"
 
Read
====

Output field values for the entity(ies) which match the specified criteria::

    grenade_cl --mode read --entity <entity> --args <args>

Where:

    * ``<entity>`` is the type of Shotgun entity to operate on (e.g., *Note*)
    * ``<args>`` is a dictionary containing a Shotgun API compatible list of search filters, and optionally, a Shotgun API compatible list of sorting rules.

Example(s)::

    $ grenade_cl --mode read --entity Note --args "{'filters':[['id', 'is', 1000]]}"
    $ grenade_cl --mode read --entity Shot --args "{'filters':[['code', 'contains', '21a']], 'order':[{'field_name':'code', 'direction':'asc'}]}"

Update
======

Update selected fields on an existing Shotgun entity::

    grenade_cl --mode update --entity <entity> --args <args>

Where:

    * ``<entity>`` is the type of Shotgun entity to operate on (e.g., *Note*)
    * ``<args>`` is a dictionary of name:value pairs describing the entity fields to update, and their new values. The id of the entity to update must be included.

Example(s)::

    $ grenade_cl --mode update --entity Note --args "{'id':1000, 'content':'updated test', 'note_links':[{'Scene':[['id', 'is', 1]]}]}"

Delete
======

Delete the Shotgun entity(ies) which match the specified criteria::

    grenade_cl --mode delete --entity <entity> --args <args>

Where:

    * ``<entity>`` is the type of Shotgun entity to operate on (e.g., *Note*)
    * ``<args>`` is a Shotgun API compatible list of search filters.

Example(s)::

    $ grenade_cl --mode delete --entity Note --args "[['id', 'is', 1000]]"
