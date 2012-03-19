.. highlight:: bash

.. _miki.user.configuring:

Configuring 
===========

When creating a documentation :ref:`structure <miki.user.setup>` using Miki, a ``conf.py`` is automatically created in 
the docs root. By default this configuration file just imports the global configuration from Miki that provides 
sensible defaults for configuring the documentation. Typically, you should not need to adjust the configuration, but
if the need arises then you can do so by simply setting new values underneath the import directive.

For example::

    # conf.py

    # Load centralised configuration
    # ---------------------------------------------------
    from miki.sphinx_config import *
    
    # Add any overrides required below
    # ---------------------------------------------------
    
    # Add directories in documentation source tree that should be excluded from build
    exclude_trees.append("./temp")
    
    
Here we have added an entry to the ``exclude_trees`` configuration value. 


Advanced
--------

A few values are derived from others, such as ``project_safe_name`` derived from ``project``. To avoid having to 
manually compute the derivatives if you want to just change ``project`` you can import the derived values setup function
and re-run it with your overrides::

    # conf.py

    # Load centralised configuration
    # ---------------------------------------------------
    from miki.sphinx_config import *
    
    # Re-setup derived values manually with a few custom overrides
    # ---------------------------------------------------------
    globals().update(compute_derived_values(project="Custom Project Name"))
    
    # Continue with normal overrides below
    # ---------------------------------------------------
    