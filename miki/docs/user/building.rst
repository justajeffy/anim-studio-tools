.. highlight:: bash

.. _miki.user.building:

Building
=========

Now that the required structure is present, let's go ahead and run the first build of your documentation. It won't say
much at present, but that's ok::

    $ miki build

    Project directory is: /drd/users/martin.phillips/identities/drd/dev/Miki
    Scanning source directory: /drd/users/martin.phillips/identities/drd/dev/Miki/sources
    Looking for Python packages...
    Added package name 'miki' to common prefixes.
    Adding defaults to configuration: {'project': 'Miki', 'modindex_common_prefix': ['miki']}
    Building html in /drd/users/martin.phillips/identities/drd/dev/Miki/docs/_build/html
    Build successful - results can be found in /drd/users/martin.phillips/identities/drd/dev/Miki/docs/_build
        
You should see some output similar to the above and you will notice that the outputs for the build can be found in the
*_build* directory in your docs directory. 

.. note::

    Don't forget to add the _build directory to your source control ignore directive so you don't check built docs
    into source control accidentally!
    
If you look in the _build directory you should see different a sub-folder for the html output.
Go ahead and open up :file:`docs/_build/html/index.html` in a browser to see your documentation.

You can build other targets by passing the :option:`--target <miki build -t>` flag to the builder. For example to build html and pdf output::

    $ miki build --target pdf --target html

Miki is mainly silent during the build process and will only show warnings and errors encountered once the build of each
target completes. If you would like to see even more information output after the build just add the :option:`--loud <miki common -l>` flag to
the command line::
    
    $ miki build --loud 
    

Cleaning
--------

By default a build will use cached information to speed up the build time. Sometimes, this information can become stale
and the results output may not be as expected. If this happens simply pass the :option:`--clean <miki build -c>` command line option to 
clean the output before building::
    
    $ miki build --clean 
    
    