.. _miki.tech.sphinx_config:

************************************************
Sphinx Configuration (``miki.sphinx_config``)
************************************************

.. automodule:: miki.sphinx_config

.. function:: setup_values 
    
    Executed by Miki (not Sphinx!). This file acts as a site wide configuration file for all documentation built with
    Miki. It defines numerous options which can be configured slightly by a dictionary of defaults passed in.
    Once called the function copies its locals into the module namespace ready for importing by a project's local 
    ``conf.py`` using::
    
        from miki.sphinx_config import *
       
    This setup allows a project to override values in a local configuration file while still providing reasonable
    defaults customised to the project (e.g. the project name can be automatically computed).
        
    :param defaults: used by some arguments to allow for possibility of overriding default value.
    
    .. warning:: 
         
         Must be called by Miki builder before Sphinx.
    
    
Further Reading
===============
     
.. seealso:: `Sphinx Configuration Values <http://sphinx.pocoo.org/config.html>`_