#                 Dr. D Studios - Software Disclaimer
#
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#
###############################################################################
"""
This module will help TD's to convert nuke animation curve to maya or houdini the
other way around also.

.. note::

    Please make sure you are running in proper kipNuke environment

.. warning::

    Dont import this module as standalone , use this module with kip project

"""
__authors__ = ["kurian.os"]
__version__ = "$Revision: 104963 $".split()[1]
__revision__ = __version__
__date__ = "$Date:  July 19, 2011 12:00:00 PM$".split()[1]
__copyright__ = "2011"
__license__ = "Copyright 2011 Dr D Studios Pty Limited"
__contact__ = "kurian.os@drdstudios.com"
__status__ = "Development"



import os
import traceback
import re
#import nuke
from rodin import logging
import kip.kip_reader as kip_reader
import node_curves as knob_curves
from kip.kip_napalm_class import *
from kip.template import *
from kip.utils.kipError import *

rodin_logger = logging.get_logger('kipNuke')
napalm_func = Napalm()

class NukeWriter():
    """
    Creating nuke curve writer class

    *Parents:*

        None

    *Children:*

        * :func:`writeOutCurves`

    """
    def __init__(self):
        """
        Base init function for nuke convert.write Class.
        """
        rodin_logger.info("kip nuke writter is initialized")
        self.precision = 3
        self.nuke_tan_types = {1:"flat", 3:"linear", 0:"spline", 18:"fixed", \
                                    4:"clamped", 16:"spline", 2:"spline"}

        self.nuke_version = "nuke,%s " % nuke.NUKE_VERSION_STRING
        self.kip_nuke_version = "kipNuke%s" % os.getenv("DRD_KIPNUKE_VERSION")

    def writeOutCurves(self, nap_file_name = None, map_node_list = None,
                        nuke_node_attribute = [], write_xml = False, left_eye = [],
                        right_eye = [],map_file_name = None):
        """

        This function will create a curve class object first and then it will write out the napalm file.In silent mode, unsupported curve types will be replaced with a default curve type of user.

        .. warning::

            If you are unable to write out napalm file or write_status=False that means napalm failed to write out.

        :param  nap_file_name: User must pass a file where he want to write out curves and make sure you supply a .nap or .xml file format(strict)

        :type nap_file_name: string

        :param map_node_list: This parm can be a dict of attribute map or a list of objects

        :type map_node_list: dict or list

        :param nuke_node_attribute: if you want to replace attribute from the map file then you can specify the override attribute here

        :type nuke_node_attribute: list

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

            >>> import nuke
            >>> import kip_nuke.convert as knb
            >>> reload(knb)
            >>> knr = knc.NukeWriter()
            >>> status,nap_file,map_file=knbw.writeOutCurves("/tmp/nuke_kip_test.nap",map_node_list=["Sphere1","Sphere2"])

        """
        if nap_file_name:
            knob_curve = knob_curves.NodeCurves()

            get_all_curves = knob_curve.getCurves(nuke_nodes_curves = map_node_list, \
                            nuke_attribute_curves = nuke_node_attribute, \
                            left_eye_curves = left_eye, right_eye_curves = right_eye)

            if write_xml:
                if not nap_file_name.endswith(".xml"):
                    split_base_ext = os.path.splitext(nap_file_name)
                    if split_base_ext[-1]:
                        nap_file_name = "%s/.xml" % (split_base_ext[0])
                    else:
                        nap_file_name = "%s/.xml" % (nap_file_name)
            else:
                if not nap_file_name.endswith(".nap"):
                    raise KipBaseError("Unknown file extension found in %s !"%nap_file_name)

            write_status, map_file, nap_file = napalm_func.write(nap_file_name, get_all_curves, \
                        debug = True, map_file_name = map_file_name, software = self.nuke_version, \
                        app_version = self.kip_nuke_version)

            rodin_logger.info("%s %s %s" % (write_status, map_file, nap_file))
            return (write_status, map_file, nap_file)
        else:
            raise KipBaseError("Expected napalm file name for write curve !")
class NukeReader():
    """

    Creating nuke curve reader class

    *Parents:*

        None

    *Children:*

        * :func:`nukeSetAttr`

    """
    def __init__(self):
        """
        Base init function for nuke convert.write Class.
        """
        rodin_logger.info("kip nuke reader is initialized")

        self.nuke_in_tan_types = {"linear":2, "spline":0, "step":1, \
                            "clamped":4, "fixed":2, "bezier":4, "cubic":0}

        self.nuke_out_tan_types = {"step":1, "linear":2}
        self.curve_attr_temp = {"X":0, "Y":1, "Z":2}
        self.precision = 3
        self.user_interpolation = nuke.SMOOTH

    def nukeSetAttr(self, nap_file_name = None, nuke_node_attribute = None, nuke_nodes = [],
                        stereo=False, left_eys = [], right_eys = [], map_file_name = None,
                        offset_value = 0, start_frame = None, end_frame = None,
                        attribute_map = None):
        """
        This function will get all curve data from a map and channel file then those data will be applied to proper nodes

        :param  nap_file_name: User must pass a file where he want to write out curves and make sure you supply a .nap or .xml file format

        :type nap_file_name: string

        :param nuke_nodes: list of nuke objects

        :type nuke_nodes: list

        :param nuke_node_attribute: if you want to replace attribute from the map file then you can specify the override attribute here

        :type nuke_node_attribute: string

        :param map_file_name: Filepath of napalm channel data

        :type map_file_name: string

        :param left_eyes: Left eye objects

        :type left_eyes: list

        :param right_eyes: Right eye objects

        :type right_eyes: list

        :param stereo: This will enable stereo curve apply for nodes

        :type stereo: bool

        :param offset_value: Animation key offset value

        :type offset_value: int

        :param start_frame: start frame to capture

        :type start_frame: int

        :param end_frame: end frame to capture

        :type end_frame: int

        :param attribute_map: This a template object from template module

        :type attribute_map: list of tuple

        Example

            >>> import nuke
            >>> import kip_nuke.convert as knc
            >>> reload(knc)
            >>> knr = knc.NukeReader()
            >>> import kip.template as template
            >>> attr_mp = template.KipTemplates()
            >>> attr_mp.ATTRMAP={"o1.ottr_1":"Transform1.rotate","o1.ottr_2":"Transform1.motionblur","o1.ottr_2":"Transform1.scale","o1.ottr_3":"Transform1.skew","o1.ottr_4":"Transform1.shutter"}
            >>> a = attr_mp.ATTRMAP
            >>> knr.nukeSetAttr(nap_file_name="/tmp/single_maya_test.nap",nuke_nodes=["Transform1"],attribute_map=a)
            >>> # Stereo transfer
            >>> knr.nukeSetAttr(nap_file_name="/tmp/maya_kip_test_s.nap",nuke_nodes=["Sphere1"],stereo=True)
            >>> # Object based transfer
            >>> knr.nukeSetAttr(nap_file_name="/tmp/maya_kip_test.nap",nuke_nodes=["Sphere1"])

        """

        if nap_file_name and os.path.exists(nap_file_name):

            if not map_file_name:
                map_file_name = kip_reader.build_map_file_name(nap_file_name)
            header_info = kip_reader.header(map_file_name)
            array_index = kip_reader.find_software_index(header_info["client_software"])

            nuke_node_list = nuke_nodes
            knob_read = kip_reader.ReadCurve()
            get_curve_class = knob_read.getCurves(nap_file_name = nap_file_name, \
                                map_file_name = map_file_name, offset_value = offset_value)

            curve_attr_index = 0

            if nuke_nodes or attribute_map:
                if stereo:
                    eye_dict_main = {}
                    if not left_eys or right_eyes:
                        curve_count = len(get_curve_class)
                        for each_e_curve in get_curve_class:
                            eye_dict_main.update({each_e_curve[0]:each_e_curve[1]})

                    curve_count = len(get_curve_class)
                    object_count = len(nuke_node_list)
                    for cnt in range(0, object_count):
                        right_curve = get_curve_class[cnt]
                        left_curve = get_curve_class[cnt+1]
                        view_dict_key = {"right":right_curve, "left":left_curve}
                        for each_curve in view_dict_key:
                            base_curve_control = view_dict_key[each_curve][-1]
                            for each_sub_curve in base_curve_control:
                                curve_type		= each_sub_curve[0]
                                curve_attr		= each_sub_curve[1][-1]
                                current_key_dict = each_sub_curve[-1]
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
                                    if attribute_map:
                                        temp_attr_keys = attribute_map.keys()
                                        for each_template in temp_attr_keys:
                                            source_details = each_template.split(".")
                                            current_node_attr = "%s.%s" % (curent_source_node, \
                                                                            each_curve[1])
                                            if current_node_attr == each_template:
                                                destenation_details = attribute_map[each_template].\
                                                                                        split(".")
                                                curve_type = destenation_details[1]
                                                current_node_name = destenation_details[0]
                                                current_node_name = nuke.toNode(current_node_name)
                                                break
                                    else:
                                        current_node_name = nuke_node_list[cnt]
                                        current_node_name = nuke.toNode(current_node_name)

                                    if curve_attr in self.curve_attr_temp.keys():
                                        curve_attr_index = self.curve_attr_temp[curve_attr]

                                    try:
                                        list_all_knobs = current_node_name.knobs()
                                        if curve_type.lower() in list_all_knobs:
                                            current_knob = current_node_name[curve_type.lower()]
                                            current_knob.splitView(each_curve)
                                            nuke_anim_curve = current_knob.setAnimated\
                                                                        (view=each_curve)
                                            nuke_anim_curve = current_knob.animations\
                                                        (view=each_curve)[curve_attr_index]

                                            for each_key in time_keys:

                                                if start_frame and end_frame:
                                                    if each_key in range(start_frame, \
                                                                            end_frame + 1):
                                                        key_index = time_keys.index(each_key)
                                                    else:
                                                        print "%s not in range not applying the"\
                                                                                " key" % each_key
                                                        continue
                                                else:
                                                    key_index = time_keys.index(each_key)
                                                current_key = nuke_anim_curve.setKey(time_keys\
                                                            [key_index], key_value[key_index])

                                            keys_from_node = nuke_anim_curve.keys()
                                            self.apply_key_controls(keys_from_node, time_keys, \
                                                key_value, in_angle, out_angle, in_weight, \
                                                out_weight, in_tan_type, out_tan_type, \
                                                in_slope, out_slope, nuke_anim_curve, \
                                                start_frame = start_frame,\
                                                end_frame = end_frame)

                                    except AttributeError:
                                        traceback.print_exc()
                                        raise KipAttrubiteError("Animation transfer failed")
                                except IndexError:
                                    print "%s dont have %s in this % index!" % \
                                        (current_node_name.name(),curve_attr,curve_attr_index)
                                    pass
                    return True

                for each_node in get_curve_class:
                    node_key = get_curve_class.index(each_node)
                    current_node_curve = each_node[2]
                    curent_source_node =  each_node[0]

                    for each_curve in current_node_curve:
                        current_key_dict = each_curve[2]
                        curve_type		= each_curve[0]
                        curve_attr		= each_curve[1][-1]
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
                        if nuke_node_attribute:
                            curve_type = nuke_node_attribute
                        else:
                            if attribute_map:
                                temp_attr_keys = attribute_map.keys()
                                for each_template in temp_attr_keys:
                                    source_details = each_template.split(".")
                                    current_node_attr = "%s.%s" % (curent_source_node, \
                                                                        each_curve[1])

                                    if current_node_attr == each_template:
                                        destenation_details = attribute_map[each_template].\
                                                                                split(".")

                                        curve_type = destenation_details[1]
                                        current_node_name = destenation_details[0]
                                        current_nuke_node = nuke.toNode(current_node_name)
                                        break
                            else:
                                current_node_name = nuke_node_list[node_key]
                                current_nuke_node = nuke.toNode(current_node_name)

                        if curve_attr in self.curve_attr_temp.keys():
                            curve_attr_index = self.curve_attr_temp[curve_attr]
                        try:
                            list_all_knobs = current_nuke_node.knobs()
                            if curve_type.lower() in list_all_knobs:
                                nuke_anim_curve = current_nuke_node[curve_type.lower()]\
                                                                        .setAnimated()

                                nuke_anim_curve = current_nuke_node[curve_type.lower()]\
                                                        .animations()[curve_attr_index]

                                for each_key in time_keys:
                                    if start_frame and end_frame:
                                        if each_key in range(start_frame, end_frame + 1):
                                            key_index = time_keys.index(each_key)
                                        else:
                                            print "%s not in range not applying the key" % each_key
                                            continue
                                    else:
                                        key_index = time_keys.index(each_key)
                                    current_key = nuke_anim_curve.setKey(time_keys[key_index], \
                                                                        key_value[key_index])

                                keys_from_node = nuke_anim_curve.keys()
                                self.apply_key_controls(keys_from_node, time_keys, key_value, \
                                in_angle, out_angle, in_weight, out_weight, in_tan_type, \
                                out_tan_type, in_slope, out_slope, nuke_anim_curve, \
                                start_frame = start_frame, end_frame = end_frame)

                        except AttributeError:
                            traceback.print_exc()
                            raise KipAttrubiteError("Animation transfer failed")
            else:
                raise KipBaseError("No nodes passed for applying the keyframe")
        rodin_logger.info("Aniamtion curve trasfer is finished !")
        return True

    def apply_key_controls(self, keys_from_node, time_keys, key_value, in_angle,
                            out_angle, in_weight, out_weight, in_tan_type,
                            out_tan_type, in_slope, out_slope, nuke_anim_curve,
                            start_frame = None, end_frame = None):

        """
        This function will apply all tangent,slop all options to keys.

        :param keys_from_node: All keys from the current animation curve

        :type keys_from_node: list

        :param time_keys: All key frame time values

        :type time_keys: list

        :param key_value: All key frame values

        :type key_value: list

        :param in_angle: all in-angle values for the current curve

        :type in_angle: list

        :param out_angle:all out-angle values for the current curve

        :type out_angle: list

        :param in_weight: all in-weight values for the current curve

        :type in_weight: list

        :param out_weight: all out-angle values for the current curve

        :type out_weight: list

        :param in_tan_type: all in-tangent type values for the current curve

        :type in_tan_type: list

        :param out_tan_type: all out-tangent type values for the current curve

        :type out_tan_type: list

        :param in_slope: all in-slop values for the current curve

        :type in_slope: list

        :param out_slope: all out-slop values for the current curve

        :type out_slope: list

        :param nuke_anim_curve: Current animation curve

        :type nuke_anim_curve: nuke anim curve

        :param start_frame: start frame to capture

        :type start_frame: int

        :param end_frame: end frame to capture

        :type end_frame: int

        Example

            >>> apply_key_controls([1,2,3],[1,2,3],[0.5,1,1.5],[0.01,0.02,0.03],[0.03,0.02,0.01],[0.01,0.02,0.03],[0.03,0.02,0.01],["spline","fixed","linear"],["linear","fixed","smooth"],[0.5,1,1.5],[0.01,0.02,0.03],xform_knob.x)

        """
        i = 0
        for key in time_keys:
            if start_frame and end_frame:
                if key in range(start_frame, end_frame + 1):
                    key_index = time_keys.index(key)
                else:
                    print "%s not in range not applying the key" % key
                    continue
            else:
                key_index = time_keys.index(key)

            if in_weight[key_index] != 1 and i != 0:
                la = in_weight[key_index] / ((keys_from_node[i].x - \
                                keys_from_node[i-1].x) / self.precision)

                if la > self.precision:
                    la = self.precision
                keys_from_node[i].la = la

            if out_weight[key_index] != 1 and i != len(time_keys)-1:
                ra = out_weight[key_index] / ((keys_from_node[i+1].x - \
                                    keys_from_node[i].x) / self.precision)

                if ra > self.precision:
                    ra = self.precision
                keys_from_node[i].ra = ra

            interpolation = self.user_interpolation
            if in_tan_type[key_index] in  self.nuke_in_tan_types.keys():
                interpolation = self.nuke_in_tan_types[in_tan_type[key_index]]

            extrapolation = None
            if out_tan_type[key_index] in  self.nuke_out_tan_types.keys():
                extrapolation = self.nuke_out_tan_types[out_tan_type[key_index]]

            current_key_obj = keys_from_node[i]

            nuke_anim_curve.changeInterpolation([current_key_obj], interpolation)

            if extrapolation:
                nuke_anim_curve.changeInterpolation([current_key_obj], extrapolation)

            keys_from_node[i].rslope = out_slope[key_index]
            keys_from_node[i].lslope = in_slope[key_index]

    def splitViewList(self, knob):

        """
        This function will check if a knob is splited or not.

        :param knob: nuke knobe object

        :type knob: knob

        :return: Split views

        :rtype: default,left,right

        Example

            >>> import nuke
            >>> import re
            >>> node = nuke.selectedNode()
            >>> knob = node["translate"]
            >>> a = splitViewList(knob)
            >>> print a
            >>> # Result:
            >>> ['default', 'left', 'right']
        """

        fullViewsString = r'default\s\{|' + '|'.join([r'%s\s\{' % v for v in nuke.views()])
        viewsRE = re.compile('(%s)' % fullViewsString)
        knobViews = [match.rstrip(' {') for match in viewsRE.findall(knob.toScript())]
        return (knobViews)

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

