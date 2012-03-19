#                 Dr. D Studios - Software Disclaimer
#
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#
###############################################################################

"""
This module will help TD's to convert maya animation curve to nuke or houdini the
other way around also.

.. note::

    Please make sure you are running in proper kipMaya environment

.. warning::

    Dont import this module as standalone , use this module with kip project

"""

__authors__ = ["kurian.os"]
__version__ = "$Revision: 105535 $".split()[1]
__revision__ = __version__
__date__ = "$Date:  July 19, 2011 12:00:00 PM$".split()[1]
__copyright__ = "2011"
__license__ = "Copyright 2011 Dr D Studios Pty Limited"
__contact__ = "kurian.os@drdstudios.com"
__status__ = "Development"



import math
import traceback
import os
import maya.cmds as cmds
import node_curves as node_curves
import kip.kip_reader as kip_reader
reload(node_curves)
from rodin import logging
from kip.kip_curve_class import *
from kip.kip_napalm_class import *
from kip.utils.kipError import *

rodin_logger = logging.get_logger('kipMaya')
napalm_func = Napalm()

class MayaWriter(object):
    """
    Creating maya curve writer class

    *Parents:*

        None

    *Children:*

        * :func:`writeOutCurves`

    """
    def __init__(self):
        """
        Base init function for maya convert.write Class.
        """
        self.supported_tan_types = ["spline", "linear", "clamped", "step", "fixed"]
        rodin_logger.info("kip maya writing class initialized")
        self.maya_version = "maya,%s" % cmds.about(v=True)
        self.kip_maya_version = "kipMaya%s" % os.getenv("DRD_KIPMAYA_VERSION")

    def writeOutCurves(self, nap_file_name=None, maya_node_attribute = [], start_frame = None,
                        end_frame=None, write_xml = False, silent = False, left_eyes = [],
                        right_eyes = [], map_file_name =None , map_node_list = None):
        """

        This function will create a curve class object first and then it will write out the
        napalm file.In silent mode, unsupported curve types will be replaced with a default
        curve type of user.

        .. warning::

            If you are unable to write out napalm file or write_status=False that means napalm
            failed to write out.

        :param  nap_file_name: User must pass a file where he want to write out curves and make
            sure you supply a .nap or .xml file format(strict)

        :type nap_file_name: string

        :param maya_node_attribute: if you want to replace attribute from the map file then you
            can specify the override attribute here

        :type maya_node_attribute: list

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

        :param map_node_list: This parm can be a dict of attribute map or a list of objects

        :type map_node_list: dict or list

        :return: Status,channel file , map file

        :rtype: boot,string,string

        Example

            >>> import kip_maya.convert as km
            >>> reload(km)
            >>> kmc = km.MayaWriter()
            >>> import kip.template as template
            >>> attr_mp = template.KipTemplates()
            >>> attr_mp.ATTRMAP={"o1.ottr_1":"t1.cutatt1","o1.ottr_2":"t1.cutatt2","o1.ottr_3":"t2.cutatt1","o1.ottr_4":"t2.cutatt2"}
            >>> a_m = attr_mp.ATTRMAP
            >>> status,nap_file,map_file = kmc.writeOutCurves(nap_file_name="/tmp/single_maya_test.nap",map_node_list=a_m,maya_node_attribute=["cutatt1","cutatt2"])
            >>> status,nap_file,map_file=kmc.writeOutCurves(nap_file_name = "/tmp/maya_kip_test_s.nap",map_node_list = ["pSphere1","pSphere2"],map_file_name= "/tmp/maya_kip_test_m.nap",left_eyes=["pSphere1"],right_eyes=["pSphere2"])

        """
        if nap_file_name:
            node_curve = node_curves.NodeCurves()
            get_all_curves = node_curve.getCurves(maya_attribute_curves = maya_node_attribute, \
                                        start_frame = start_frame, end_frame = end_frame, \
                                        silent = silent, left_eye_curves = left_eyes, \
                                        right_eye_curves = right_eyes, \
                                        maya_nodes_curves = map_node_list)
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
                                                debug = True,map_file_name = map_file_name, \
                                                software = self.maya_version, \
                                                app_version = self.kip_maya_version)

            rodin_logger.info("%s %s %s" % (write_status, map_file, nap_file))
            del(map_node_list, maya_node_attribute, left_eyes, right_eyes)
            return (write_status, map_file, nap_file)
        else:
            raise KipBaseError("Expected napalm file name for write curve !")

class MayaReader(object):
    """

    Creating maya curve reader class

    *Parents:*

        None

    *Children:*

        * :func:`mayaSetAttr`

    """
    def __init__(self):
        """
        Base init function for maya convert.write Class.
        """
        rodin_logger.info("kip maya read class initialized")

    def mayaSetAttr(self, nap_file_name = None, maya_nodes = [], maya_node_attribute = None,
                            map_file_name = None, offset_value = 0, start_frame = None,
                            end_frame = None, attribute_map = None):
        """
        This function will get all curve data from a map and channel file then those data will be applied to proper nodes

        :param  nap_file_name: User must pass a file where he want to write out curves and make sure you supply a .nap or .xml file format

        :type nap_file_name: string

        :param maya_nodes: list of maya objects

        :type maya_nodes: list

        :param maya_node_attribute: if you want to replace attribute from the map file then you can specify the override attribute here

        :type maya_node_attribute: string

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

            >>> import kip_maya.convert as km
            >>> reload(km)
            >>> kpr=km.MayaReader()
            >>> import kip.template as template
            >>> attr_mp = template.KipTemplates()
            >>> attr_mp.ATTRMAP={"t1.cutatt1":"o1.ottr_1","t1.cutatt2":"o1.ottr_2","t2.cutatt1":"o1.ottr_3","t2.cutatt2":"o1.ottr_4"}
            >>> a = attr_mp.ATTRMAP
            >>> kpr.mayaSetAttr(nap_file_name = "/tmp/single_maya_test.nap",maya_nodes=["o1"],attribute_map=a)
            >>> # Object based transfer
            >>> kpr.mayaSetAttr(nap_file_name = "/tmp/maya_kip_test.nap",maya_nodes=["pSphere2"])

        """
        if nap_file_name:

            if not map_file_name:
                map_file_name = kip_reader.build_map_file_name(nap_file_name)
            header_info = kip_reader.header(map_file_name)
            array_index = kip_reader.find_software_index(header_info["client_software"])

            maya_node_list = maya_nodes
            knob_read = kip_reader.ReadCurve()
            get_curve_class = knob_read.getCurves(nap_file_name = nap_file_name, \
                                map_file_name = map_file_name, offset_value = offset_value)

            if maya_node_list:
                for each_node in get_curve_class:
                    node_key = get_curve_class.index(each_node)
                    current_node_curve = each_node[2]
                    curent_source_node = each_node[0]
                    for each_curve in current_node_curve:
                        curve_attr        = each_curve[1]
                        current_key_dict = each_curve[2]
                        time_keys         = current_key_dict["time"]
                        key_value        = current_key_dict["key_value"]
                        in_angle        = current_key_dict["in_angle"]
                        out_angle        = current_key_dict["out_angle"]
                        in_weight        = current_key_dict["in_weight"]
                        out_weight        = current_key_dict["out_weight"]
                        in_tan_type        = current_key_dict["in_tan_type"]
                        out_tan_type    = current_key_dict["out_tan_type"]

                        if maya_node_attribute:
                            curve_attr = maya_node_attribute
                        else:
                            if attribute_map:
                                current_maya_node = None
                                temp_attr_keys = attribute_map.keys()
                                for each_template in temp_attr_keys:
                                    source_details = each_template.split(".")
                                    current_node_attr = "%s.%s" % (curent_source_node, curve_attr)
                                    if current_node_attr == each_template:
                                        destenation_details = attribute_map[each_template].split\
                                                                                            (".")
                                        curve_attr = destenation_details[1]
                                        current_maya_node = destenation_details[0]
                                        break
                                if not current_maya_node:
                                    break
                            else:
                                current_maya_node = maya_node_list[node_key]

                        if not cmds.attributeQuery(curve_attr, node = current_maya_node, \
                                                                            exists = True):
                            continue

                        for time in time_keys:
                            if start_frame and end_frame:
                                if time in range(start_frame, end_frame + 1):
                                    key_index = time_keys.index(time)
                                else:
                                    print "%s not in range not applying the key" % time
                                    continue
                            else:
                                key_index = time_keys.index(time)
                            key_curve = cmds.setKeyframe(current_maya_node, \
                                    v=key_value[key_index], at=curve_attr, time=time )

                            cmds.keyTangent(current_maya_node, edit=True, time = (time, time), \
                                            attribute = curve_attr, ia = in_angle[key_index], \
                                            oa = out_angle[key_index], iw=  in_weight[key_index], \
                                            ow = out_weight[key_index])
            else:
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

