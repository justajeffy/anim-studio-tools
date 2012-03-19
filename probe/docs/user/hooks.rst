.. _probe-user-hooks:

Hooks
=====

Python provides a number of hooks to change the default behaviour of the 
interpreter.  This could range from except hooks used when an uncaught 
exception is raise to import hooks that affect the way different modules are 
imported.

.. _probe-user-hooks-except:

Uncaught Exception Handler
--------------------------

When an exception is raised and uncaught, the interpreter calls sys.excepthook 
In an interactive session this happens just before control is returned to the 
prompt; in a Python program this happens just before the program exits.  An
alternative handler is provided that, when used in a Python program, dumps out
to a pdb session in post mortem mode instead of exiting.

.. code-block:: bash
    
    $ python2.5 -c "import probe.hooks.except_; probe.hooks.except_.init(); raise Exception('This is an uncaught exception')"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    Exception: This is an uncaught exception
    
    > <string>(1)<module>()
    (Pdb)
    
