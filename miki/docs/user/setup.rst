.. highlight:: bash

.. _miki.user.setup:

Setting Up
===========

To get up and running quickly, Miki provides some utility functions that allow you to create the required 
documentation structure for your project.

From your project's root directory simply run ``miki structure --create``::

    $ cd myproject
    $ miki structure --create
    
This will attempt to create the required documentation structure under your project's root directory. Typically this
will result in a structure similar to the following:

.. image:: /static/structure.png


You may receive warnings stating that structure already exists and new structure can't be created. This might be
because you already existing files and folders created. You can pass an :option:`--update <miki structure -u>` flag to Miki to enable the 
overwriting of files in this case::

    $ miki structure --update

.. note::
    
    No files or folders will ever be deleted even when using the update flag. 
    
    
Automatically Generating Technical Documentation
-------------------------------------------------

Miki is based on the principle of useful hand-written documentation rather than generic API generated documentation.
However, it if often useful to also include auto generated documentation for technical reference. You can
do this by hand by using the ``autodoc`` directive (explained further in :ref:`miki.user.editing.autodoc`). 

To help get you and up running, Miki also includes functionality to automatically generate documentation templates for 
all of your existing Python source files. To take advantage of this as part of your documentation setup simply run
``miki autodoc``::
 
    $ miki autodoc
 
This will generate in the set output directory (docs/technical by default) .rst files for all your Python source files.
Each .rst file will include the appropriate autodoc directives. So that you can edit these files and still use the autodoc
command to add new files Miki will by default not overwrite any files when using this method. If you do want to overwrite
files use the :option:`--update <miki autodoc -u>` flag::

    $ miki autodoc --update
    
Lastly, if you want to change the default output directory specify the :option:`--output-directory <miki autodoc -d>` flag::

    $ miki autodoc --output-directory technical/api
    
