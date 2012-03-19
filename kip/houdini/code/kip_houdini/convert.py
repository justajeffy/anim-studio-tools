#                 Dr. D Studios - Software Disclaimer
#
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#
###############################################################################
"""
This module will help TD's to convert houdini animation curve to nuke or maya
the other way around also.

.. note::

    Please make sure you are running in proper kipHoudini environment

.. warning::

    Dont import this module as standalone , use this module with kip project

"""

__authors__ = ["kurian.os"]
__version__ = "$Revision: 104961 $".split()[1]
__revision__ = __version__
__date__ = "$Date:  July 19, 2011 12:00:00 PM$".split()[1]
__copyright__ = "2011"
__license__ = "Copyright 2011 Dr D Studios Pty Limited"
__contact__ = "kurian.os@drdstudios.com"
__status__ = "Development"


import os
import traceback
#import hou
import napalm.core as nap_core
import node_curves as node_curves
import kip.kip_reader as kip_reader
reload(node_curves)
reload(kip_reader)
from rodin import logging
from kip.kip_curve_class import *
from kip.kip_napalm_class import *
from kip.utils.kipError import *
from kip.template import *


rodin_logger = logging.get_logger('kipHoudini')
napalm_func = Napalm()

GLOBAL_FPS = 24
GLOBAL_TIME = 1


class HoudiniWriter(object):

    """
    Creating houdini curve writer class

    *Parents:*

        None

    *Children:*

        * :func:`writeOutCurves`

    """

    def __init__(self):
        """
        Base init function for houdini convert.write Class.
        """
        rodin_logger.info("kip houdini writing class initialized")
        self.houdini_version = "houdini,%s" % hou.applicationVersionString()
        self.kip_houdini_version = "kipHoudini%s" % os.getenv("DRD_KIPHOUDINI_VERSION")

    def writeOutCurves(self, nap_file_name = None , houdini_nodes = [],
                            houdini_node_attributes = [], start_frame = None,
                            end_frame = None, write_xml = False, silent = False,
                            left_eyes = [], right_eyes = [], map_file_name = None):

        """
        This function will create a curve class object first and then it will write out the napalm file.

        .. warning::

            If you are unable to write out napalm file or write_status=False that means napalm failed to write out.

        :param  nap_file_name: User must pass a file where he want to write out curves and make sure you supply a .nap or .xml file format(strict)

        :type nap_file_name: string

        :param houdini_nodes: list of houdini objects(strict)

        :type houdini_nodes: list

        :param houdini_node_attribute: if you want to replace attribute from the map file then you can specify the override attribute here

        :type houdini_node_attribute: list

        :param start_frame: start frame to capture

        :type start_frame: int

        :param end_frame: end frame to capture

        :type end_frame: int

        :param write_xml: If you want to write out a xml file instead of napalm file then this should be true

        :type end_frame: string

        :param left_eyes: Left eye objects

        :type left_eyes: list

        :param right_eyes: Right eye objects

        :type right_eyes: list

        :param map_file_name: Filepath of napalm channel data

        :type map_file_name: string

        :return: Status,channel file , map file

        :rtype: boot,string,string

        Example

            >>> import kip_houdini.convert as kh
            >>> reload(kh)
            >>> khcw = kh.HoudiniWriter()
            >>> status,nap_file,map_file=khcw.writeOutCurves(nap_file_name = "/tmp/houdini_kip_test_s.nap",map_file_name= "/tmp/houdini_kip_test_m.nap",houdini_nodes = ["/obj/geo/xform_1","/obj/geo/xform_2"],left_eyes=["/obj/geo/xform_1"],right_eyes=["/obj/geo/xform_2"])

        """
        if nap_file_name:
            node_curv = node_curves.NodeCurves()
            get_all_curves = node_curv.getCurves(houdini_node_curves = houdini_nodes, \
                                houdini_attribute_curves = houdini_node_attributes, \
                                start_frame = start_frame, end_frame = end_frame, \
                                silent = silent, left_eye_curves = left_eyes, \
                                right_eye_curves = right_eyes)
            if write_xml:
                if not nap_file_name.endswith(".xml"):
                    split_base_ext = os.path.splitext(nap_file_name)
                    if split_base_ext[-1]:
                        nap_file_name = "%s/.xml" % (split_base_ext[0])
                    else:
                        nap_file_name = "%s/.xml" % (nap_file_name)
            else:
                if not nap_file_name.endswith(".nap"):
                    raise KipBaseError("Unknown file extension found in %s !" % nap_file_name)

            write_status, map_file, nap_file = napalm_func.write(nap_file_name, get_all_curves, \
                                        debug = True, map_file_name = map_file_name, \
                                        software = self.houdini_version, \
                                        app_version = self.kip_houdini_version)

            rodin_logger.info("%s %s %s" % (write_status, map_file, nap_file))
            return (write_status, map_file, nap_file)
        else:
            raise KipBaseError("Expected napalm file name for write curve !")

class HoudiniReader(object):
    """

    Creating houdini curve reader class

    *Parents:*

        None

    *Children:*

        * :func:`houSetAttr`

    """
    def __init__(self):
        """
        Base init function for houdini convert.write Class.
        """
        rodin_logger.info("kip houdini read class initialized")
        self.nuke_tan_types = {"spline":"spline()", "linear":"linear()", \
                                    "constant":"constant()", "cubic":"bezier()"}

        self.channel_match = {'translateX':'tx', 'translateY':'ty', 'translateZ':'tz', \
                                'rotateX':'rx', 'rotateY':'ry', 'rotateZ':'rz', \
                                'scaleX':'sx', 'scaleY':'sy', 'scaleZ':'sz'}

    def houSetAttr(self, nap_file_name = None, houdini_nodes = [], houdini_node_attribute = None,
                            map_file_name = None, offset_value = 0, start_frame = None,
                            end_frame = None, attribute_map = None):
        """
        This function will get all curve data from a map and channel file then those data will be applied to proper nodes

        :param  nap_file_name: User must pass a file where he want to write out curves and make sure you supply a .nap or .xml file format

        :type nap_file_name: string

        :param houdini_nodes: list of houdini objects

        :type houdini_nodes: list

        :param houdini_node_attribute: if you want to replace attribute from the map file then you can specify the override attribute here

        :type houdini_node_attribute: string

        :param map_file_name: Filepath of napalm channel data

        :type map_file_name: string

        :param offset_value: Animation key offset value

        :type offset_value: int

        :param start_frame: start frame to capture

        :type start_frame: int

        :param end_frame: end frame to capture

        :type end_frame: int

        :param attribute_map: This a template object from template module

        :type attribute_map: list of tuple

        Example

            >>> import kip_houdini.convert as kh
            >>> reload(kh)
            >>> khpr=kh.HoudiniReader()
            >>> import kip.template as template
            >>> attr_mp = template.KipTemplates()
            >>> attr_mp.ATTRMAP={"t1.cutatt1":"/obj/geo1/xform1.ottr_1","t1.cutatt2":"/obj/geo1/xform1.ottr_2","t2.cutatt1":"/obj/geo1/xform1.ottr_3","t2.cutatt2":"/obj/geo1/xform1.ottr_4"}
            >>> a = attr_mp.ATTRMAP
            >>> khpr.houSetAttr(nap_file_name="/tmp/single_maya_test.nap",houdini_nodes="/obj/geo1/xform1",attribute_map=a)

        """
        if nap_file_name:

            if not map_file_name:
                map_file_name = kip_reader.build_map_file_name(nap_file_name)
            header_info = kip_reader.header(map_file_name)
            array_index = kip_reader.find_software_index(header_info["client_software"])

            houdini_node_list = houdini_nodes
            knob_read = kip_reader.ReadCurve()
            get_curve_class = knob_read.getCurves(nap_file_name = nap_file_name, \
                                map_file_name = map_file_name, offset_value = offset_value)

        for each_node in get_curve_class:
            node_key = get_curve_class.index(each_node)
            current_node_curve = each_node[2]
            curent_source_node = each_node[0]
            for each_curve in current_node_curve:
                curve_attr		= each_curve[1]
                current_key_dict = each_curve[2]
                time_keys 		= current_key_dict["time"]
                key_value		= current_key_dict["key_value"]
                in_angle		= current_key_dict["in_angle"]
                out_angle		= current_key_dict["out_angle"]
                in_weight		= current_key_dict["in_weight"]
                out_weight		= current_key_dict["out_weight"]
                in_tan_type		= current_key_dict["in_tan_type"]
                out_tan_type	= current_key_dict["out_tan_type"]
                in_slope		= current_key_dict["in_slope"]
                out_slope		= current_key_dict["out_slope"]
                try:
                    for time in time_keys:

                        if houdini_node_attribute:
                            curve_attr = houdini_node_attribute
                        else:
                            if attribute_map:
                                temp_attr_keys = attribute_map.keys()
                                for each_template in temp_attr_keys:
                                    source_details = each_template.split(".")
                                    current_node_attr = "%s.%s" % (curent_source_node, \
                                                                        each_curve[1])
                                    if current_node_attr == each_template:
                                        destenation_details = attribute_map[each_template]\
                                                                                .split(".")
                                        curve_attr = destenation_details[1]
                                        current_houdini_node = destenation_details[0]
                                        current_houdini_node = hou.node(current_houdini_node)
                                        break
                            else:
                                current_houdini_node = hou.node(houdini_node_list[node_key])
                        if start_frame and end_frame:
                            if time in range(start_frame, end_frame + 1):
                                key_index = time_keys.index(time)
                            else:
                                print "%s not in range not applying the key" % time
                                continue
                        else:
                            key_index = time_keys.index(time)
                        in_tan_v = in_tan_type[key_index]
                        if self.nuke_tan_types.has_key(in_tan_v):
                            in_tan_v = self.nuke_tan_types[in_tan_v]
                        else:
                            in_tan_v = "bezier()"
                        hkey = hou.Keyframe()
                        hkey.setTime((time_keys[key_index]/GLOBAL_FPS))
                        hkey.setValue(key_value[key_index])
                        hkey.setExpression("bezier()")
                        hkey.setExpression("spline()")
                        hkey.setInAccel(in_weight[key_index])
                        hkey.setAccel(out_weight[key_index])
                        hkey.setInSlope(in_slope[key_index])
                        hkey.setSlope(out_slope[key_index])
                        this_node_attr = curve_attr
                        if self.channel_match.has_key(curve_attr):
                            this_node_attr = self.channel_match[curve_attr]
                        hou_nod = current_houdini_node.parm(this_node_attr).setKeyframe(hkey)
                except:
                    traceback.print_exc()
                    raise KipBaseError("No objects found in node list!")
        rodin_logger.info("Aniamtion curve trasfer is finished !")
        return True

def header(map_file_name):
    """

    This function will return a dict of header details from map file

    :param map_file_name: Filepath of napalm channel data

    :type map_file_name: string

    :return: header details

    :rtype: dict

    """
    if os.path.exists(map_file_name):
        nap_header = kip_reader.header(map_file_name)
        return nap_header
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

