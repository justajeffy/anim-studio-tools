.. _probe-user-fixtures:

Fixtures and Mock Objects
=========================

When writing and running unit tests, it is not always possible to have access
to a clean instance of some resource for testing - for example Tank and Shotgun.
This makes unit testing difficult as you potentially have inconsistent data to
test with and so can not rely on the results.

Probe provides a number of 'mock' objects that simulate a resource such as Tank
or Shotgun so the surrounding code can be used in unit tests more effectively.
Currently there are mock objects of Shotgun, Tank, Rodin Log and Fedex Database
objects.

.. _probe-user-fixtures-shotgun:

Mock Shotgun
------------

This is used to mock a Shotgun connection object (the result of ``shotgun_v3.connect(...)``).

.. _probe-user-fixtures-log:

Mock Log
--------

This is used to mock a Rodin log object (the result of ``rodin.logging.get_logger(...)``).

.. _probe-user-fixtures-tank:

Mock Tank
---------

This is used to mock Tank.

.. _probe-user-fixtures-database:

Mock Database
-------------

This is used to mock an SQLAlchemy database connection, primarily used by Fedex.
