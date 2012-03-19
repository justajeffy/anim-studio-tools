.. _miki.tech.command_line:

************************************************
Command Line Reference (``miki``)
************************************************

.. program:: miki

**Usage:** miki [build,structure,autodoc] <options>

.. note:: Miki expects to be run in the root folder of your project.

Required
=========

Use the first argument to specify the mode by specifying one of the following three directives:

.. cmdoption:: build
    
    Build the documentation.
    
.. cmdoption:: structure
    
    Manage documentation structure
    
.. cmdoption:: autodoc
    
    Automatically generate reference files for all Python sources.
      

Common Arguments
==================

.. program:: miki common 

.. cmdoption:: -h, --help

    Get help on a particular mode. E.g. ``miki -h`` or ``miki build -h``

.. cmdoption:: -l, --loud

    When set causes more information to be output to the screen during operations.

.. cmdoption:: -p, --project-directory

    The project root directory containing all sources and docs. Defaults to current location.
    
.. cmdoption:: -s, --source-directory

    A directory containing source files for the project. Can be relative to project directory. 
    Can be specified more than once. Default is 'sources'.
        

Build Mode
===========

.. program:: miki build

.. cmdoption:: -n <name>, --project-name <name>
    
    The name of the project. Auto-calculated if not set.

.. cmdoption:: -r <release>, --project-release <release>
    
    The project release string - e.g. 1.2.0. Auto-calculated if not set.

.. cmdoption:: -v <version>, --project-version <version>
    
    The project version string - e.g. 1.0. Auto-calculated if not set.

.. cmdoption:: -t <type>, --target <type>
    
    A target to build. For example 'html' (which is the default).

.. cmdoption:: -c, --clean
    
    When set will remove any existing output and cached data before building.
      
.. cmdoption:: -d <file>, --doxyfile <file>
    
    Location of doxyfile. Can be relative to project directory.
      

Structure Mode
===============

.. program:: miki structure

.. cmdoption:: -c, --create
    
    Create documentation structure based on a central template.
    
.. cmdoption:: -u, --update
    
    Update existing documentation structure to match central template. Will overwrite, but not remove existing files and folders.
        
.. cmdoption:: -v, --validate
    
    Validate existing documentation structure. Will report missing files and folders.
        

Autodoc Mode
=============

.. program:: miki autodoc

.. cmdoption:: -d <directory>, --output-directory <directory>
    
    Directory to output generated files in. Can be relative to docs directory. Default is 'technical'.
    
.. cmdoption:: -u, --update
    
    If set will overwrite, but not delete existing files.
    

Calling From Python
====================

You can execute the main program directly from Python by importing Miki and passing a list of command line options
to the main function::

    import miki
    miki.main(arguments)
    
.. module:: miki.__init__
    
.. autofunction:: miki.main
   
  

