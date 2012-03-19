.. _miki.user.editing:

Editing 
========

If you look in your docs folder you will find that Miki auto-created (during :ref:`setup <miki.user.setup>`) some default files and a rough structure 
for separating documentation into User and Technical documentation. 

Table Of Contents 
-----------------

The key to how your documentation is structured is through the use of table of contents directives at various levels.
Together all these TOC's make up your documentation tree.

If you look at the root :file:`index.rst` you will see it simply defines a table of contents for the user and technical 
sections using the ``.. toctree::`` directive::

    .. toctree::
    
        user/index
        technical/index
        
The two entries refer to sub-directories :file:`./user/index.rst` and :file:`./technical/index.rst`. 


Adding Pages
------------

To add a page to the documentation there are a few actions to perform:

#. Create a *.rst* file for your new page in the appropriate place (you might want to copy and paste an existing file).
   For example, create ``introduction.rst`` in the ``user`` sub-directory.
#. Edit the file to contain at least a header (see `Headings`_). 

   .. warning::
        
       Without a header, a link to your file will not appear even if you specify it in a table of contents.

#. Now edit the :file:`index.rst` at the same level as your new file adding the name of your file, minus the extension, to 
   the table of contents directive. Continuing the example, edit :file:`user/index.rst` to have::
        
        .. toctree::
            
            introduction

#. :ref:`Build <miki.user.building>` your documentation and you should see your new page appear as an entry in the 
   *User Documentation* table of contents.


Adding Parts
-------------

Adding an entirely new part is similar to adding files, but with an extra step to create an index.rst with an 
appropriate table of contents directive.

#. Create a new subfolder for your new part at the appropriate place. For example, create an ``api`` directory under
   the technical directory. 
#. Now, create a new :file:`index.rst` file in your new subfolder. For example, create ``technical/api/index.rst``.
   (see :ref:`index.rst <miki.user.templates.index_rst>` for a template to use).
#. Finally, edit the :file:`index.rst` in the directory **above** your new subfolder (e.g. ``technical/index.rst``), adding the 
   name of your subfolder index file to the table of contents directive::
           
        .. toctree::
            
            api/index
            

Headings
---------

Headings are used to further break up your documentation into sections, sub-sections etc. Use the following guide for
which heading format to use where:

* % with overline for the highest level (the master heading of your documentation)
* \# with overline for parts
* \* with overline for chapters
* = for sections
* \- for sub-sections
* ^ for sub-sub-sections
   
(see :ref:`Templates <miki.user.templates.headings>` for more explicit examples).


.. _miki.user.editing.cross_referencing:


Cross Referencing 
-----------------

You can link to sections using traditional restructured text links. 

To create a link to the Headings section above we could use the following::

    `Headings`_
    
which would render as `Headings`_. To change the link text use the format::

    `Link Text <Headings>`_

which gives you `Link Text <Headings>`_

However, if you want to make your cross referencing maintainable and also work across files then you should use labels 
and the ``:ref:`` directive.

A label takes the form ``.. _my_reference_label:`` and when placed immediately before a heading allows you to reference
the heading via the label. If you look at the source for this page you will see that a label has been placed before
the heading for this section::

   .. _miki.user.editing.cross_referencing:

    Cross Referencing 
    -----------------
    
We can then link to the section using the ref directive::
 
    :ref:`miki.user.editing.cross_referencing` or :ref:`Link text <miki.user.editing.cross_referencing>`

which would appear as :ref:`miki.user.editing.cross_referencing` or :ref:`Link text <miki.user.editing.cross_referencing>`

.. note::

    The ``:ref:`` directive does not use the leading underscore '_' in the label name.
    
This makes your documentation more resilient to change - if you change the heading text your links will automatically
update, whereas using the traditional syntax would require you to update your links. Additionally, other files in 
your document can reference the heading via the label.
 
Please try to use the syntax of dots '.' for separating folders and files and use underscores '_' for replacing
spaces when creating labels.


Referencing External Documents 
------------------------------

As well as referencing other parts of your documentation through labels, you may wish to also provide links to other 
documentation. To do this you could just include a link using the syntax ```Link Text <href>`_``::

`Drd Studios <http://drdstudios.com.au/>`_ 

which renders as `Drd Studios <http://drdstudios.com.au/>`_.
    
However, this can become difficult to manage if links change over time - particularly if you are referencing other Miki
documentation that is versioned. To help with this you can place in your config file a list of links and then refer to 
them throughout your documentation with the ``.. extref::`` directive::

    # conf.py
    # ---------------
    external_links = {"Drd": "http://drdstudios.com.au" }
    

A link such as::

`:extref:`Drd Studios <Drd>`

would then render as :extref:`Drd Studios <Drd>`
            
If the reference text contains a forward slash it will be split and the first part used as reference, with the 
second part appended to the generated link::

    :extref:`Contact Drd <Drd/content/contact-us>`

would expand to http://drdstudios.com.au/content/contact-us as you can see if you click :extref:`Contact Drd <Drd/content/contact-us>`

Images
-------

If you want to include images in your documentation you should place them in the existing ``static`` folder under the
docs root. You can then reference them in your text using the ``image`` directive::

    .. image:: /static/example.png

would generate:

    .. image:: /static/example.png

You can add options beneath the directive to specify size etc::

    .. image:: /static/example.png
        :width: 80px
        

.. image:: /static/example.png
    :width: 80px


Diagrams
---------

It is also possible to include basic diagrams in your documentation. These are generated using a Graphviz plugin which
is enabled by default. Here is a simple example to wet your appetite, but for more info read up on the 
:extref:`graph syntax <Sphinx/ext/graphviz.html>`::

    .. graph:: project_triangle
        
       graph[
           splines = false,
       ];
       "good" -- "fast";
       "good" -- "cheap";
       "cheap" -- "fast" [constraint=false];

which renders as 

.. graph:: project_triangle
    
    graph[
           splines = false,
    ];
    "good" -- "fast";
    "good" -- "cheap";
    "cheap" -- "fast" [constraint=false];


For Python source code you can also have inheritance diagrams automatically generated by using the ``.. inheritance-diagram::``
directive::
 
    .. inheritance-diagram:: python.module
    
For example, to generate an inheritance diagram for the Miki error classes we can use::

    .. inheritance-diagram:: miki.errors

which outputs

.. inheritance-diagram:: miki.errors


Source Code
-----------

It is often useful to reference source code, give code examples and provide some kind of API directory in your 
documentation. To create code examples simply use the ``.. code-block:: language`` directive::

    .. code-block:: python
    
        x = "a string"
        print x

The above would render as:

.. code-block:: python

    x = "a string"
    print x

You can also specify other languages, for example ``.. code-block:: ruby`` would highlight Ruby code.


.. _miki.user.editing.autodoc:

Auto-documenting
^^^^^^^^^^^^^^^^^

It is also possible to extract comments and definitions from source code automatically - this is useful when providing
an API reference document.

For Python code you can use the `autodoc <http://sphinx.pocoo.org/ext/autodoc.html>`_ directives. You need to tell Miki 
where to find your source code so it can inspect them. Use the :option:`--source-directory <miki common -s>` flag for this 
(the default is a folder called "sources" found relative to the project directory).

An auto-documentation example:

.. code-block:: python

    .. automodule:: miki.builder
        :members:
        
Which would appear like the :ref:`Builder Reference <miki.tech.builder>` page in Miki's technical documentation. 


Auto-listing
^^^^^^^^^^^^^

When you have a lot of code described in one file it can be useful to provide the reader with a summary listing of the
code on the page. You can do this automatically using the ``.. autolisting::`` directive:

.. code-block:: python

    .. autolisting::
    
    .. automodule:: miki.autodoc
        :members:

would generate html similar to the following

.. image:: /static/autolisting_example.png
    :width: 600px
  

Further Reading
----------------
    
.. seealso:: :extref:`RestructuredText Primer <Sphinx/rest.html>`
