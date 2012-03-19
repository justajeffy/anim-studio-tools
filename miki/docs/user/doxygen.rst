.. highlight:: bash

.. _miki.user.doxygen:

Doxygen Integration
===================

For those that already have documentation written for Doxygen, this section will cover how to use that existing 
documentation with Miki without having to rewrite it all.


Doxyfile
----------

Doxygen requires a Doxyfile to be present that specifies various options for the Doxygen build. Miki will use that same
file when it builds your Doxygen docs, but will override a few options to ensure that XML is output to a directory Miki
knows about.


Building
----------- 

To build Doxygen documentation with Miki, simply provide Miki with the location of your Doxyfile:: 

    $ miki build --doxyfile ./mydoxyfile.dox
    
.. note::

    By default Miki looks for a file named ``Doxyfile`` directly under the project directory.
    

Referencing
------------

To use the Doxygen output in your documentation you need to reference it. You can do this using special directives. To
reference the Doxygen output in its entirity simply add ``.. doxygenindex::`` to a ``.rst`` file where you want 
the Doxygen documentation to appear. You can also reference specific functions using the ``.. doxygenfunction:: <function name>``
directive. (See :ref:`Doxygen Template <miki.user.templates.doxygen>`)


Further Reading
----------------

.. seealso:: `Doxygen Integration Plugin (Breathe) <http://michaeljones.github.com/breathe/index.html>`_
 