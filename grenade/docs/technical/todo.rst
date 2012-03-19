***********
Future Work
***********

:Release: |version|
:Date: |today|

Here is an overview of identified areas of improvement for future versions of
Grenade:

    * Implement translators for each entity type (it doesn't look like we are able to create a one-size fits all/generic translator)
    * Update model to make a pass of all multi_entity properties at create or delete time when a translator is attached, to ensure that the values assigned to those fields have been translated (.append() issue again)
