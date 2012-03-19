.. _probe-user-decorators:

Decorators
==========

Probe provides some decorators to make debugging Python scripts easier.  These
are described here.

.. _probe-user-decorators-timer:

Timer
-----

The timer decorator provides a simple way to measure the execution time of a 
function call.  If you are testing a function for performance, you can ask for
the same function to be called multiple times to obtain an average or standard
deviation.  For example::

    >>> from probe.decorators.timer import timer
    >>> @timer(iterations=1, output_directory="/tmp")
    >>> def func():
    ...     # do work
    ...     # return
    ...
    >>> func()

The duration in seconds of each iteration is written to a csv file found in the
specified output directory.

.. _probe-user-decorators-strace:

Stack Trace
-----------

The stack trace decorator allows you to run an ``strace`` command on the 
currently executing code.  For example::

    >>> from probe.decorators.strace import strace
    >>> @strace(follow_children=True, output_directory="/tmp")
    >>> def func():
    ...     # do work
    ...     # return
    ...
    >>> func()

The output of the strace will be written to a file in the specified output 
directory.

.. warning::
    When using this decorator with a function that calls Tank code, please 
    ensure that the ``follow_children`` parameter is set to ``False``.  This 
    is because Tank requires escalated permissions when publishing new 
    revisions which the strace call is unable to follow correctly (as it is 
    still running as the current user).

.. _probe-user-decorators-profile:

Profile
-------

The profile decorator uses the cProfile module to gather statistics about the
wrapped function.  These are then written to a file in the specified output 
directory.  For example::

    >>> from probe.decorators.profile import profile
    >>> @profile(graph=True, output_directory="/tmp")
    >>> def func():
    ...     # do work
    ...     # return
    ...
    >>> func()

With ``graph`` set to ``True``, a png image showing the execution callgraph will
also be generated.

.. warning::
    This decorator redirects stdout temporarily.  If you are using threads, and all
    threads haven't completed before the wrapped function returns, anything written 
    to stdout by these threads will most likely corrupt the results gathering.  So,
    either don't use this decorator with threads, or make sure they have all completed
    before the wrapped function returns.
