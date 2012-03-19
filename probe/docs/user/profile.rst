.. _probe-user-profile:

Profile
=======

.. _probe-user-profile-gprof2dot:

Generating a Callgraph Using cProfile
-------------------------------------

Probe provides a handy alias which you can use to automatically generate a 
callgraph for a Python script using the cProfiler.  This uses a third party
module called :extref:`gprof2dot<GProf2Dot>`.

For example, to generate a callgraph for a script called ``leaky.py``, from a 
Probe environment you would run::

    $ cProfile leaky.py

This would call your script, using cProfile to gather detailed profiling 
information, and after a (hopefully) short pause, open up xnview with the out.
It will look something like the image shown below.  Deatils about what the nodes
and edges mean can be found in the :extref:`gprof2dot documentation<GProf2DotOutput>`.

.. image:: /static/callgraph.png
    :alt: Callgraph Generate from cProfile Output
    :align: center
    :width: 550px

.. note::
    That stats file from CProfile will be saved under ``/tmp/output.pstats`` 
    where it can be used for further debugging.


.. _probe-user-profile-kcachegrind:

Profiling your program using kcachegrind
----------------------------------------

Probe also provides another alias to profile your program using
:extref:`kcachegrind<KCacheGrind>`. kcachegrind is a linux utility
which uses the same sort of cProfile/calltree profile statistics to
show a more interactive means of viewing your profiled application.

kcachegrind is *not* installed by default on our linux systems. You
can find it in the 'kdesdk' package. Add kdesdk via yum, or use
puppet::

    iWant { "kdesdk": arch => x86_64 }

To get data from a python run into kcachegrind, we use another
third-party module called
:extref:`pyprof2calltree<PyProf2Calltree>`. (pyprof2calltree is
installed via drd-config and required by probe, so you will pick it up
automagically when you do something like::

    $ drd-env probe myPackage

To run kcachegrind (assuming you've now got it installed) on your
python script called ``slow.py``, from an environment containing Probe
and the module(s) you need to run::

    $ kcgProfile slow.py

Like the cProfile alias above, this calls your script, using the
cProfile module to gather detailed profiling 
information, and then opens up kcachegrind with your profiled data.

It will look something like the image shown below. kcachegrind is
somewhat complicated and can be an overload of data, but is a lot
more interactive than just viewing the
static call graph. (And docs indicate it should provide that same
call graph functionality, but i haven't had luck getting this to show up from my
python profiling.) Check the docs for more details, but playing with
it for a bit shows it's fairly intuitive, and can be a good tool for
honing in on performance problems.

.. image:: /static/kcachegrind.png
    :alt: kcachegrind screenshot
    :align: center
    :width: 550px

If the profiling is too broad, you can modify the code you want to
profile. import cProfile directly at the spots you wish to
profile instead of profiling the entire thing::

    >> import cProfile
    >> code = "my_instance = MyClass(); my_instance.do_something()"
    >> cProfile.run(code, '/tmp/myProfile.pstats')

and then from a shell::

    $ pyprof2calltree -i /tmp/myProfile.pstats -o
    /tmp/myProfile.kgrind.001
    $ kcachegrind /tmp/myProfile.kgrind.001

pyprof2calltree has a -k option to run kcachegrind directly, but think
this had some problems. It too has a python interface, but this
definitely had some problems. According to the docs, you should be
able to do this::

    >> import cProfile
    >> code = "my_instance = MyClass(); my_instance.do_something()"
    >> profiler = cProfile.Profile()
    >> profiler.runctx(code, locals(), globals())
    >> stats = profiler.getstats()

    >> from pyprof2calltree import convert, visualize
    >> visualize(stats)                            # run kcachegrind
    >> convert(stats, '/tmp/myProfile.kgrind.002') # save for later

But didn't work for me, i gave up trying.

Of course doing profiling from just the code you care about will give
you results tailored to the code you care about, which is good, but it
also gives you the ability to more easily time the top-level
execution. Of course timing individual bits of your code is possible
(and sometimes preferable) via the ``timeit`` or ``time`` modules, but
for a quick and easy timing of the top-level execution of your code
(if it's cmd-lineable somehow), i stole this from how Arsenal jobs
run::

    /usr/bin/time --format=baztime:real:%e:user:%U:sys:%S:iowait:%w myCmdLine

See the docs for the linux time cmd (man time) -- 'user' time is
probably most relevant when profiling for speed, but you can also
query memory and swap parameters as well.
    
    
.. note::
    That stats file from CProfile will be saved under ``/tmp/kcg_output.pstats``
    where it can be used for further debugging.

.. note::
    I had a problem running kcachegrind in some environments --
    notably if maya2009 was in there, it had some libgcc problems
    (old stdc++ library used by maya probably)
