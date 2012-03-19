.. highlight:: bash

.. _miki.user.templates:

Templates
=========

The following are a series of templates you can use when creating new files or structure. Note that you can also look
at existing documentation and click the ``Show Source`` link in the left nav to find out how it is done.

.. _miki.user.templates.headings:

Headings
---------

.. code-block:: rest

    ##################
    Part
    ##################
    
    ******************
    Chapter
    ******************
    
    Section
    ==================
    
    Sub-Section
    ------------------
    
    Sub-Sub-Section
    ^^^^^^^^^^^^^^^^^^
    
    
    
.. _miki.user.templates.index_rst:

index.rst
---------

.. code-block:: rest

    ##################
    Part Heading
    ##################
    
    :Release: |version|
    :Date: |today|
    
    Short overview string for part
    
    .. toctree::
        :maxdepth: 2
            
        page_1
        page_2
        sub_part_1/index


.. _miki.user.templates.doxygen:

Doxygen
---------

.. code-block:: rest

    Doxygen Reference Title
    ========================
    
    :Release: |version|
    :Date: |today|
    
    Short overview string.
    
    .. doxygenindex::
    
    