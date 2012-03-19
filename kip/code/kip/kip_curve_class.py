###############################################################################

"""
This module is written for creating curve,key and object class.kip_curve_class module depends
on 3 main classes :class:`ClassObject`, :class:`CurveClass` and :class:`CurveKeyClass`.
This module not depend on any external libs

.. note::
    This module should be expanded now its written as a base class so its need more structured
    re writing maybe.

variable list::

    0 -- CURVE_NAME_KEY
    1 -- CURVE_PARM_NAME_KEY
    2 -- ANGLULAR_UNITS

"""
__authors__ = ["kurian.os"]
__version__ = "$Revision: 104960 $".split()[1]
__revision__ = __version__
__date__ = "$Date:  July 19, 2011 12:00:00 PM$".split()[1]
__copyright__ = "2011"
__license__ = "Copyright 2011 Dr D Studios Pty Limited"
__contact__ = "kurian.os@drdstudios.com"
__status__ = "Development"


CURVE_NAME_KEY = "name"
CURVE_PARM_NAME_KEY = "parm"
ANGLULAR_UNITS = "radians"


class ClassObject:
    """
    Object Class:
        This class iterate through each curve from the list and find animation values from the curve and return it as a stucture of array.

    *Parents:*

        None

    *Children:*

        * :func:`output`
    """
    def __init__(self):
        """
        Base variable list ( animation is expecting a list value always)

        variable list::

            0 -- name
            1 -- type
            2 -- animation

        """
        self.name = None
        self.type = None
        self.animation = []

    def output(self):
        """
        This function will take all inputs from the `init` and it will go through each curve then return a stucture of array.

        :return: stucture of array

        :rtype: array

        Example

            >>> from kip.kip_curve_class import *
            >>> kipCurveClassObject = ClassObject()
            >>> kipCurveClassObject.name="test"
            >>> kipCurveClassObject.type="transform"
            >>> kipCurveClassObject.animation.append(kipCurveObject)# this var always looking for a CurveObject
            >>> a = kipCurveClassObject.output()

        """
        py_curve_objects = [self.name, self.type]
        for curve in self.animation:
            py_curve_objects.append(curve.output())

        return py_curve_objects


class CurveClass:
    """
    Curve Class:
        This class will iterate through each key and then it will build a curve class object from the key values.

    *Parents:*

        None

    *Children:*

        * :func:`add_key`

        * :func:`output`

    """
    def __init__(self):
        """
        Base variable list (all variable is expecting a list value always)

        variable list::

            0 -- name
            1 -- parm
            2 -- parm_attr
            3 -- keys
            4 -- curve_keys # not needed until you want to return it
        """
        self.name = []
        self.parm = []
        self.parm_attr = []
        self.keys = []
        self.curve_keys = []

    def add_key(self):
        """
        This function will return current key values as a list

        :return: key values

        :rtype: list

        """
        return self.curve_keys

    def output(self):
        """
        This function will return the curve class object built from key values.

        :return: array or structured

        :rtype: list

        Example

            >>> from kip.kip_curve_class import *
            >>> kipCurveObject = CurveClass()
            >>> kipCurveObject.keys.append(key)
            >>> kipCurveObject.parm.append("rotation")
            >>> kipCurveObject.parm_attr.append(rotationX)
            >>> a = kipCurveObject.output()#we wont use this way normally.
        """
        curve_out_list = []
        for key in self.keys:
            curve_list = []
            curve_key_struct = {}
            current_key_index = self.keys.index(key)
            curve_list.append(self.parm[current_key_index])
            curve_list.append(self.parm_attr[current_key_index])
            current_key = key.output()
            #~ print self.parm_attr[current_key_index],current_key
            curve_temp_dict = current_key.keys()
            for each_dict_key in curve_temp_dict:
                if not curve_key_struct.has_key(each_dict_key):
                    curve_key_struct[each_dict_key] = ""
                    curve_key_struct[each_dict_key] = current_key[each_dict_key]
                else:
                    curve_key_struct[each_dict_key] = current_key[each_dict_key]
            curve_list.append(curve_key_struct)
            curve_out_list.append(curve_list)
        return curve_out_list


class CurveKeyClass:
    """
    Key Class:
        This class will creat a key class from user input.

    *Parents:*

        None

    *Children:*

        * :func:`output`

    """
    def __init__(self):
        """
        Basic init funtion and here is input variable list and all variable is expecting a list value always

        variable list::

            0 -- in_tan_type
            1 -- out_tan_type
            2 -- time
            3 -- value
            4 -- in_angle
            5 -- in_weight
            6 -- out_angle
            7 -- out_weight
            8 -- in_slope
            9 -- out_slope

        """
        self.in_tan_type = []
        self.out_tan_type = []
        self.time = []
        self.value = []
        self.in_angle = []
        self.in_weight = []
        self.out_angle = []
        self.out_weight = []
        self.in_slope = []
        self.out_slope = []

    def output(self):

        """
        This function will out put all key values as a single key ( dict )

        :return: key

        :rtype: dict

        Example

            >>> from kip.kip_curve_class import *
            >>> key = CurveKeyClass()
            >>> kipCurveObject = CurveClass()
            >>> key.time.append(time[i])
            >>> key.value.append(value[i])
            >>> key.in_angle.append(in_angle[i])
            >>> key.out_angle.append(out_angle[i])
            >>> key.in_weight.append(in_weight[i])
            >>> key.out_weight.append(out_weight[i])
            >>> etc..
            >>> kipCurveObject.keys.append(key)

        """

        key_dict = {}
        key_dict.update({"time":self.time, "key_value":self.value, "in_weight":self.in_weight, \
        "out_weight":self.out_weight, "in_angle":self.in_angle, "out_angle":self.out_angle, \
        "in_tan_type":self.in_tan_type, "out_tan_type":self.out_tan_type, \
        "in_slope":self.in_slope, "out_slope":self.out_slope})
        return key_dict

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

