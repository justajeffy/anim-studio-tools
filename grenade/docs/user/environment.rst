*********************
Environment Variables
*********************

:Release: |version|
:Date: |today|

In order to operate successfully, Grenade requires Shotgun connection details.
By default, Grenade will connect to the Dr. D Shotgun sandbox, as the RnD test
script user. These settings may be overridden via the following environment
variables:

    * ``GRENADE_SG_HOST`` - hostname of the Shotgun service to connect to (e.g., *http://shotgun-sandbox*)
    * ``GRENADE_SG_USER`` - Shotgun script username to connect as (e.g., *rnd*)
    * ``GRENADE_SG_SKEY`` - Shotgun script key to use (e.g., *5d5d12f1b...*)

You should override these settings within your environment as necessary to 
connect to the correct Shotgun instance.
