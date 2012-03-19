.. _probe-user-memory:

Memory
======

.. _probe-user-memory-objgraph:

Using Objgraph
--------------

Probe includes the :extref:`objgraph<Objgraph>` module for inspecting objects in
memory and reference counts.  It can be imported as::

    >>> import objgraph

For further documentation, consult the :extref:`online documentation<Objgraph>`.

.. _probe-user-memory-sizeof:

Using Sizeof
------------

Probe includes the :extref:`ActiveState Recipe 546530<ActiveState/546530>` for 
calculating the size of Python objects.  It can be imported as::

    >>> import sizeof

For further documentation, consult :extref:`ActiveState<ActiveState/546530>`.

.. _probe-user-memory-guppy_local:

Profile The Memory Usage of A Local Process
-------------------------------------------

Finding information about the memory usage of a Python session is relatively 
trivial using :extref:`Guppy<Guppy/#Documentatio>`::

    >>> from guppy import hpy
    >>> hp=hpy()
    >>> print hp.heap()
    Partition of a set of 355556 objects. Total size = 50878704 bytes.
     Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
         0   9605   3 15908984  31  15908984  31 dict (no owner)
         1  88502  25  6958496  14  22867480  45 str
         2  99496  28  6804200  13  29671680  58 tuple
         3      1   0  4630248   9  34301928  67 array.array
         4   3935   1  4123880   8  38425808  76 dict of
                                                 tank.local._transform_rule_component.TransformRuleCompo
                                                 nent
         5  94799  27  2275176   4  40700984  80 int
         6  11668   3  1367520   3  42068504  83 list
         7   4048   1  1133440   2  43201944  85 dict of tank.local._vfs_tokens.ConstantToken
         8    252   0   713376   1  43915320  86 dict of module
         9    530   0   555440   1  44470760  87 dict of tank.local._propertystore.MetaData
    <276 more rows. Type e.g. '_.more' to view.>

.. _probe-user-memory-guppy_remote:

Profile The Memory Usage of A Remote Process
--------------------------------------------

It is possible to use :extref:`Guppy<Guppy/#Documentatio>` to profile the memory
usage of a remote process.  This is done using the :extref:`monitor<Guppy/heapy_Use.html#heapykinds.Use.monitor>`
functionality.

1. Before starting the process you wish to monitor, you must add the following
   import statement.  This assumes that the ``guppy`` package is available in 
   your environment::

    import guppy.heapy.RM

2. Run your process as normal

3. In another shell/python interpretor, you can import ``guppy`` and use the 
   :extref:`monitor<Guppy/heapy_Use.html#heapykinds.Use.monitor>` method to 
   begin to begin profiling::

    >>> from guppy import hpy
    >>> hp=hpy()
    >>> hp.monitor()
    <Monitor>

4. Find the CID of your running process by running the ``lc`` command.  Note 
   that this is not the same as your process (or PID) ID::

    <Monitor> lc
    CID PID   ARGV
     1 18383 ['leaky.py']

5. Connect to this process using the ``sc`` command to start an Annex shell::

    <Monitor> sc 1
    Remote connection 1. To return to Monitor, type <Ctrl-C> or .<RETURN>
    <Annex> 

6. Create a new interpreter instance::

    <Annex> int
    Remote interactive console. To return to Annex, type '-'.

7. And then you have access to the standard :extref:`heap<Guppy/heapy_Use.html#heapykinds.Use.heap>` 
   object to profile the memory usage of the remote process::

    >>> hp.heap()
    Partition of a set of 355556 objects. Total size = 50878704 bytes.
     Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
         0   9605   3 15908984  31  15908984  31 dict (no owner)
         1  88502  25  6958496  14  22867480  45 str
         2  99496  28  6804200  13  29671680  58 tuple
         3      1   0  4630248   9  34301928  67 array.array
         4   3935   1  4123880   8  38425808  76 dict of
                                                 tank.local._transform_rule_component.TransformRuleCompo
                                                 nent
         5  94799  27  2275176   4  40700984  80 int
         6  11668   3  1367520   3  42068504  83 list
         7   4048   1  1133440   2  43201944  85 dict of tank.local._vfs_tokens.ConstantToken
         8    252   0   713376   1  43915320  86 dict of module
         9    530   0   555440   1  44470760  87 dict of tank.local._propertystore.MetaData
    <276 more rows. Type e.g. '_.more' to view.>

.. seealso::
    Guppy is a powerful memory profiling toolset for Python, however it can be
    complicated to use, take some time to read the :extref:`documentation<Guppy/heapy_Use.html>`.
