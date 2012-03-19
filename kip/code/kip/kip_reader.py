#                 Dr. D Studios - Software Disclaimer
#
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#
###############################################################################
"""
This module will return curve object from a nap_file_name and map_file_name
"""
__authors__ = ["kurian.os"]
__version__ = "$Revision: 104960 $".split()[1]
__revision__ = __version__
__date__ = "$Date:  July 19, 2011 12:00:00 PM$".split()[1]
__copyright__ = "2011"
__license__ = "Copyright 2011 Dr D Studios Pty Limited"
__contact__ = "kurian.os@drdstudios.com"
__status__ = "Development"


import os
import napalm.core as nap_core
from rodin import logging
from kip.kip_napalm_class import *
from kip.kip_curve_class import *

rodin_logger = logging.get_logger('kip')
napalm_func = Napalm()

class ReadCurve():

    """

    This class will read napalm channel file and map file then it will try to find the channel data based on the map file . Those data will be re formatted as a curve class object.

    *Parents:*

        None

    *Children:*

        * :func:`getCurves`

        * :func:`read`

    """

    def __init__(self):
        """

        Base init function for Napalm Class.

        """
        rodin_logger.info("Initialized Reader Curve")

    def getCurves(self, map_file_name = None, nap_file_name = None, offset_value = 0):

        """

        This function will return the curve object based on the channel and map file

        :param nap_file_name: Filepath for writting napalm data

        :type nap_file_name: string

        :param map_file_name: Filepath for writting napalm channel data

        :type map_file_name: string

        :param offset_value: Animation key offset value

        :type offset_value: int


        Example

            >>> import kip.kip_reader as krp

            >>> kncw = krp.ReadCurve()

            >>> curve = kncw.getCurves("/tmp/maya_kip_test_s.nap",map_file_name="/tmp/maya_kip_test_m.nap")

        .. seealso::

            * :func:`read`

        .. versionchanged:: 0.0.5

            Fixed the map_file_name fixed.

        .. todo::

            More in-line comment should be added

        :return: Curve Object

        :rtype: list

        """
        curve_objects = self.read(nap_file_name = nap_file_name, map_file_name = map_file_name, \
                                                                    offset_value = offset_value)
        return curve_objects

    def read(self, nap_file_name = None, map_file_name = None, offset_value = 0):

        """

        This function will return the curve object based on the channel and map file and this
        function added for backward compatable module.

        :param nap_file_name: Filepath for reading napalm data

        :type nap_file_name: string

        :param map_file_name: Filepath for reading napalm channel data

        :type map_file_name: string

        :param offset_value: Animation key offset value

        :type offset_value: int

        .. warning::

            This function not suppose to call from any API call . use :func:`getCurves` for creating curves.


        .. seealso::

            * :func:`getCurves`

        .. versionchanged:: 0.0.5

            Fixed the map_file_name fixed.

        .. todo::

            More in-line comment should be added

        :return: Curve Object

        :rtype: list

        """

        curve_objects = []
        if nap_file_name:
            if not map_file_name:
                map_file_name = build_map_file_name(nap_file_name)
            if os.path.exists(map_file_name):
                map_table = nap_core.load(map_file_name)
                map_tbl_keys = map_table.keys()
                chan_table = nap_core.load(nap_file_name)
                for obj_cnt in  range(0, len(map_tbl_keys)):
                    kipCurveClassObject = ClassObject()
                    kipCurveObject = CurveClass()
                    if map_tbl_keys[obj_cnt] == "header":
                        continue
                    get_obj_table = map_table[map_tbl_keys[obj_cnt]]
                    obj_tbl_keys = get_obj_table.keys()
                    eye_value = get_obj_table["eye_val"]
                    for each_curv_cnt in range(0, len(obj_tbl_keys)):
                        attr_name = obj_tbl_keys[each_curv_cnt]
                        if attr_name == "eye_val":
                            continue
                        curve_dict = get_obj_table[obj_tbl_keys[each_curv_cnt]]
                        curve_type		= attr_name[:-1]
                        curve_attr		= attr_name
                        get_all_channels = napalm_func.getAllChannles(nap_file_name)
                        time_keys 		= get_all_channels[curve_dict["time"]].contents
                        key_value		= get_all_channels[curve_dict["key_value"]].contents
                        in_angle		= get_all_channels[curve_dict["in_angle"]].contents
                        out_angle		= get_all_channels[curve_dict["out_angle"]].contents
                        in_weight		= get_all_channels[curve_dict["in_weight"]].contents
                        out_weight		= get_all_channels[curve_dict["out_weight"]].contents
                        in_tan_type		= get_all_channels[curve_dict["in_tan_type"]].contents
                        out_tan_type	= get_all_channels[curve_dict["out_tan_type"]].contents
                        in_slope		= get_all_channels[curve_dict["in_slope"]].contents
                        out_slope		= get_all_channels[curve_dict["out_slope"]].contents
                        key = CurveKeyClass()
                        for time_fr in time_keys:
                            key_index = time_keys.index(time_fr)
                            key.time.append(time_keys[key_index]+offset_value)
                            key.value.append(key_value[key_index])
                            key.in_angle.append(in_angle[key_index])
                            key.out_angle.append(out_angle[key_index])
                            key.in_weight.append(in_weight[key_index])
                            key.out_weight.append(out_weight[key_index])
                            key.in_tan_type.append(in_tan_type[key_index])
                            key.out_tan_type.append(out_tan_type[key_index])
                            key.in_slope.append(in_slope[key_index])
                            key.out_slope.append(out_slope[key_index])

                        kipCurveObject.keys.append(key)
                        kipCurveObject.parm.append(curve_type)
                        kipCurveObject.parm_attr.append(curve_attr)

                    kipCurveClassObject.name = map_tbl_keys[obj_cnt]
                    kipCurveClassObject.type = eye_value
                    kipCurveClassObject.animation.append(kipCurveObject)
                    curve_objects.append(kipCurveClassObject.output())
                return curve_objects
            else:
                rodin_logger.error("%s is not exists in filesystem"%map_file_name)

def header(map_file_name):
    """

    This function will return a dict of header details from map file

    :param map_file_name: Filepath of napalm channel data

    :type map_file_name: string

    :return: header details

    :rtype: dict

    """
    map_table = nap_core.load(map_file_name)
    map_header = map_table["header"]
    return map_header

def build_map_file_name(nap_file_name):
    """

    This function will help to find the map file from a nap file

    :param map_file_name: Filepath of napalm channel data

    :type map_file_name: string

    :return: map file name

    :rtype: string

    """
    ext_spliter = os.path.splitext(os.path.basename(nap_file_name))
    get_file_ext = ext_spliter[-1]
    set_file_base = "%s_map%s" % (ext_spliter[0], get_file_ext)
    map_file_name = "%s/%s" % (os.path.dirname(nap_file_name), set_file_base)
    return map_file_name

def find_software_index(software):
    """

    This function will help to find the current software used from the software dict

    :param software: Software name string

    :type software" string

    :return: software index

    :rtype: int

    """
    array_index = None
    if software == "maya":
        array_index = 0
    elif software == "nuke":
        array_index = 1
    elif software == "houdini":
        array_index = 2
    else:
        array_index = 3

    return array_index

def build_node_tree(nap_file_name, map_file_name=None):
    """

    This function will return node.attr list from the map file

    :param nap_file_name: Filepath for reading napalm data

    :type nap_file_name: string

    :param map_file_name: Filepath for reading napalm channel data

    :type map_file_name: string

    :return: list for node.attribute

    :rtype: list

    """
    if not map_file_name:
        map_file_name = build_map_file_name(nap_file_name)
    get_node_list = nap_core.load(map_file_name)
    map_tbl_keys = get_node_list.keys()
    node_attribute_tree = []
    for each_key in map_tbl_keys:
        if each_key == "header":
            continue
        obj_tbl_keys = get_node_list[each_key].keys()
        for each_attr in obj_tbl_keys:
            if each_attr == "eye_val":
                continue
            else:
                nd_tree = "%s.%s" % (each_key, each_attr)
                node_attribute_tree.append(nd_tree)
    return node_attribute_tree

def readData(nap_file_name, map_file_name = None, attribute_map = None):
    """

    This function will return target based mapping from the source object based on the attribute map
    and if the attribute map is empty then it will return the curve object.

    :param nap_file_name: Filepath for reading napalm data

    :type nap_file_name: string

    :param map_file_name: Filepath for reading napalm channel data

    :type map_file_name: string

    :param attribute_map: This a template object from template module

    :type attribute_map: list of tuple

    :return: Curve Object

    :rtype: list

    Example
        >>> import kip.kip_reader as kr
        >>> import kip.template as template
        >>> attr_mp = template.KipTemplates()
        >>> attr_mp.ATTRMAP={"o1.ottr_1":"t1.cutatt1","o1.ottr_2":"t1.cutatt2","o1.ottr_3":"t2.cutatt1","o1.ottr_4":"t2.cutatt2"}
        >>> a_m = attr_mp.ATTRMAP
        >>> g = kr.readData("/tmp/single_maya_test.nap",attribute_map=a_m)
        >>> g
        >>> [['t1',
              'default',
              ['cutatt1',
               {'in_angle': [7.1250171661376953, 0.0, -5.7105932235717773],
                'in_slope': [0.12500001490116119, 0.0, -0.10000000149011612],
                'in_tan_type': ['clamped', 'clamped', 'clamped'],
                'in_weight': [1.0, 1.0, 1.0],
                'key_value': [0.0, 0.5, 0.0],
                'out_angle': [7.1250171661376953, 0.0, -5.7105932235717773],
                'out_slope': [0.12500001490116119, 0.0, -0.10000000149011612],
                'out_tan_type': ['clamped', 'clamped', 'clamped'],
                'out_weight': [1.0, 1.0, 1.0],
                'time': [1.0, 5.0, 10.0]}]],


    """
    if not map_file_name:
        map_file_name = build_map_file_name(nap_file_name)
    curve_cls = ReadCurve()
    return_data = []
    curve_data = curve_cls.getCurves(nap_file_name = nap_file_name, map_file_name = map_file_name)
    if attribute_map:
        for each_node in curve_data:
            node_name = each_node[0]
            eye_value = each_node[1]
            reader_attribute = []
            current_node_curve = each_node[2]
            for each_curve in current_node_curve:
                attribute_name = each_curve[1]
                current_key_dict = each_curve[2]
                time_keys         = current_key_dict["time"]
                key_value        = current_key_dict["key_value"]
                in_angle        = current_key_dict["in_angle"]
                out_angle        = current_key_dict["out_angle"]
                in_weight        = current_key_dict["in_weight"]
                out_weight        = current_key_dict["out_weight"]
                in_tan_type        = current_key_dict["in_tan_type"]
                out_tan_type        = current_key_dict["out_tan_type"]
                in_slope        = current_key_dict["in_slope"]
                out_slope        = current_key_dict["out_slope"]

                current_node_name = None
                temp_attr_keys = attribute_map.keys()
                for each_template in temp_attr_keys:
                    source_details = each_template.split(".")
                    current_node_attr = "%s.%s" % (node_name, attribute_name)
                    if current_node_attr == each_template:
                        destenation_details = attribute_map[each_template].split(".")
                        attribute_name_prs = destenation_details[-1]
                        del (destenation_details[-1])
                        current_node_prs_name = ".".join(destenation_details)
                        curve_attr = attribute_name_prs
                        current_node_name = current_node_prs_name
                        break
                if not current_node_name:
                    break

                if not current_node_name in reader_attribute:
                    reader_attribute.append(current_node_name)
                    reader_attribute.append(eye_value)
                    reader_attribute.append([])
                    return_data.append(reader_attribute)

                attribute_mixer = []
                attribute_mixer.append(curve_attr)
                attribute_dict = {"time":time_keys, "in_angle":in_angle, "in_slope":in_slope, \
                "in_tan_type":in_tan_type, "in_weight":in_weight, "key_value":key_value, \
                "out_angle":out_angle, "out_slope":out_slope, "out_tan_type":out_tan_type, \
                "out_weight":out_weight}

                attribute_mixer.append(attribute_dict)
                get_key_index = find_it(current_node_name, return_data)
                if get_key_index >= 0 :
                    return_data[get_key_index][2].append(attribute_mixer)

    else:
        return_data = curve_data
    return return_data

def find_it(key, list):
    """
    find item from a dict
    """
    for index, sublist in enumerate(list):
        try:
            if sublist[0] == key:
                return index
        except:
            return None

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

