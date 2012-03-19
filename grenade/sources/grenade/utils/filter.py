#!/usr/bin/python2.5
#                 Dr. D Studios - Software Disclaimer
#
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

__authors__ = ["Daniela Hasenbring"]
__version__   = '$Revision: 101581 $'.split()[1]
__revision__  = __version__ # For pylint
__date__      = '$Feb 9, 2010 1:54:23 PM$'.split()[1]

__copyright__ = '2011'
__license__   = "Copyright 2011 Dr D Studios Pty Limited"
__contact__   =  "daniela.hasenbring@drdstudios.com"
__status__    = "Development"
__doc__       = ""

####################################################################################

OPERATOR_AND     = 'and'
OPERATOR_OR      = 'or'

def convert(shotgun_filter, default_operator = OPERATOR_AND):
    """
    This function converts the given Filter into a low-level Shotgun API Filter to give you control over 'and'/'or'-operators.

    To use the 'or'-operator you can either pass a list of values like this;

    convert(['sg_status_1', 'is', ['apr', 'nfr']])


    Or you can pass a list of filters like this:

    convert([['entity', 'is', '...'], [ ['sg_status_1', 'is', 'apr'], ['sg_status_1', 'is', 'nfr'] ] ])


    This way you can do a query like "If 'status' is 'apr' or 'something else' is 'value'".

    You can also specify OPERATOR_AND or OPERATOR_OR as the last parameter in this list to define the Logical Operator. Default is OPERATOR_OR.

    :param shotgun_filter:
        Shotgun filter to convert to low-level API

    :param default_operator:
        The operator to use if no operator is specified in the shotgun_filter.

    :returns:
        The low-level API filter

    .. versionadded:: 1.6.0

    """
    PATH        = 0
    RELATION    = 1
    VALUES      = 2

    conditions = []
    if len(shotgun_filter) > 0:
        if isinstance(shotgun_filter[-1], str):
            default_operator = shotgun_filter[-1]
            shotgun_filter = shotgun_filter[0:-1]

        for filter in shotgun_filter:
            if len(filter) > 0:
                if not isinstance(filter[0], list):
                    if len(filter) == 3:
                        if isinstance(filter[VALUES], list):
                            filter_conditions = []
                            for value in filter[VALUES]:
                                filter_conditions.append({'path': filter[PATH], 'relation': filter[RELATION], 'values': [value]})
                            conditions.append({'logical_operator': OPERATOR_OR, 'conditions': filter_conditions})
                        else:
                            conditions.append({'path': filter[PATH], 'relation': filter[RELATION], 'values': [filter[VALUES]]})
                else:
                    conditions.append(convert(filter, OPERATOR_OR))
    return {'logical_operator': default_operator, 'conditions': conditions}


def find(session, entity, filter, fields = ['id']):
    """
    This wraps the find-function of shotgun, but first converts the filter to the low-level API.

    :param session:
        An active Shotgun session.
    :param entity:
        Entity that corresponds to the Shotgun entity type of interest.
    :param filter:
        Shotgun filters defining the entities to be found.
    :param fields:
        A list of fields to query the schema for - optimisation over returning all fields for the
        entity which can be slow and overkill in some situations.
    :returns:
        Find results from shotgun:

    .. versionadded:: 1.6.0

    """
    return session.find(entity, convert(filter), fields)

def find_one(session, entity, filter, fields = ['id']):
    """
    This wraps the find_one-function of shotgun, but first converts the filter to the low-level API.

    This wraps the find-function of shotgun, but first converts the filter to the low-level API.

    :param session:
        An active Shotgun session.
    :param entity:
        Entity that corresponds to the Shotgun entity type of interest.
    :param filter:
        Shotgun filters defining the entities to be found.
    :param fields:
        A list of fields to query the schema for - optimisation over returning all fields for the
        entity which can be slow and overkill in some situations.
    :returns:
        Find results from shotgun:

    .. versionadded:: 1.6.0

    """
    return session.find_one(entity, convert(filter), fields)

# Copyright 2008-2012 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios)
#
# This file is part of anim-studio-tools.
#
# anim-studio-tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# anim-studio-tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with anim-studio-tools.  If not, see <http://www.gnu.org/licenses/>.

