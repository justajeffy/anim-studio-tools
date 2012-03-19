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
from rodin import logging
from kip.kip_curve_class import *
from kip.utils.kipError import *
import maya.cmds as cmds

rodin_logger = logging.get_logger('kipMaya')

class NodeCurves(object):
    """
    Creating maya curve node class

    *Parents:*
        None

    *Children:*

        * :func:`getCurves`

    """
    def __init__(self):
        """
        Base init function for maya cuve create Class.
        """
        self.supported_tan_types = ["spline", "linear", "clamped", "step", "fixed"]
        rodin_logger.info("kip maya curve class initialized")

    def getCurves(self, maya_nodes_curves = [], maya_attribute_curves = [], silent = False,
                        start_frame = None, end_frame = None,
                                left_eye_curves = [], right_eye_curves = []):
        """
        This function will get animation curve from maya nodes and all anim curve data will be processed and it will generate a curve class.In silent mode, unsupported curve types will be replaced with a default curve type of user.

        :param maya_nodes: list of maya objects

        :type maya_nodes: list

        :param silent: If this option is true then we will ignore all tangent errors.

        :type silent: bool

        :param start_frame: start frame to capture

        :type start_frame: int

        :param end_frame: end frame to capture

        :type end_frame: int

        :param left_eye_curves: Left eye objects

        :type left_eye_curves: list

        :param right_eye_curves: Right eye objects

        :type right_eye_curves: list

        :return: Curve class object

        :rtype: curve

        Example

            >>> curve_obj = getCurves(maya_nodes = ["pSphere1","pSphere2"],start_frame = 10,end_frame = 100,silent = False,left_eye_curves=["pSphere1"],right_eye_curves=["pSphere2"])

        """
        temp_maya_nodes_curves = maya_nodes_curves
        curves_from_maya_node = []
        if not temp_maya_nodes_curves:
            raise KipBaseError("Exporter need object list to export data !"\
                                                " Nothing found to Export!")

        build_node_list = []
        build_node_attr_list = []
        build_dest_node_list = []
        build_accepted_attributes = []
        if type(temp_maya_nodes_curves)  == dict:
            for each_nd in temp_maya_nodes_curves.keys():
                node_name_tr, node_attr_tr = temp_maya_nodes_curves[each_nd].split(".")
                build_accepted_attributes.append(temp_maya_nodes_curves[each_nd].split(".")[1])
                dest_node_name_tr = each_nd.split(".")[0]
                build_node_list.append(node_name_tr)
                build_node_attr_list.append(node_attr_tr)
                build_dest_node_list.append({node_name_tr:dest_node_name_tr})

        elif type(temp_maya_nodes_curves) == list:
            build_node_list = temp_maya_nodes_curves
        else:
            raise KipBaseError("Unknown mapping structure found ! list and dict only supported!")
#        current_prser_index = 0
        for each_node in build_node_list:
            kipCurveClassObject = ClassObject()
            kipCurveObject = CurveClass()
            current_prser_index = build_node_list.index(each_node)
            attribute_val = build_node_attr_list[current_prser_index]
            a_curves = cmds.keyframe(each_node, name = True, query = True)
            if start_frame and end_frame:
                if isinstance(start_frame, int ) and isinstance(end_frame, int ):
                    a_curves = cmds.keyframe(each_node, name = True, \
                                time = (int(start_frame), int(end_frame)), query= True)
                else:
                    message = ("%s or %s is not a int value , non int values are not support"\
                                            " for exporting time frame"%(start_frame,end_frame))

                    raise KipBaseError(message)

            try:
                if not a_curves:
                    continue

                for each_curve in a_curves:
                    count_split = "%s.%s" % (each_node, attribute_val)
                    current_node_prs_name = each_node
                    try:
                        node_attr_type = cmds.attributeName(count_split, n=True).split(" ")[0]
                        current_attr_name = cmds.attributeName(count_split, l=True)
                    except:
                        continue
                    if maya_attribute_curves:
                        for each_pass_attr in maya_attribute_curves:
                            if current_attr_name.find(each_pass_attr) == -1:
                                continue

                    if not current_attr_name in build_accepted_attributes:
                        continue

                    value = cmds.keyframe(each_curve, absolute = True, query = True, \
                                                                    valueChange = True)
                    time = cmds.keyframe(each_curve, absolute=True, query=True, timeChange=True)
                    in_angle = cmds.keyTangent(each_curve, query=True, ia=True)
                    out_angle = cmds.keyTangent(each_curve, query=True, oa=True)
                    in_weight = cmds.keyTangent(each_curve, query=True, iw=True)
                    out_weight = cmds.keyTangent(each_curve, query=True, ow=True)
                    in_tan_type = cmds.keyTangent(each_curve, query=True, itt=True)
                    out_tan_type = cmds.keyTangent(each_curve, query=True, ott=True)
                    i = 0
                    key = CurveKeyClass()
                    for t in time:
                        key.time.append(time[i])
                        key.value.append(value[i])
                        key.in_angle.append(in_angle[i])
                        key.out_angle.append(out_angle[i])
                        key.in_weight.append(in_weight[i])
                        key.out_weight.append(out_weight[i])

                        in_tan_type_prs = str(in_tan_type[i])
                        if not in_tan_type_prs in self.supported_tan_types:
                            if silent:
                                in_tan_type_prs = "user"
                            else:
                                raise KipTangentError("%s not supported tangent to nuke, Please "\
                                "update anim curve with any one of this %s" % \
                                                        (in_tan_type_prs, self.supported_tan_types))

                        key.in_tan_type.append(in_tan_type_prs)


                        out_tan_type_prs = str(out_tan_type[i])
                        if not out_tan_type_prs in self.supported_tan_types:
                            if silent:
                                out_tan_type_prs = "user"
                            else:
                                raise KipTangentError("%s not supported tangent to nuke, Please "\
                                "update anim curve with any one of this %s"%(out_tan_type_prs, \
                                                                        self.supported_tan_types))

                        key.out_tan_type.append(out_tan_type_prs)

                        rad_in_angle = math.radians(in_angle[i])
                        rad_out_angle = math.radians(out_angle[i])
                        key.in_slope.append(math.tan(rad_in_angle))
                        key.out_slope.append(math.tan(rad_out_angle))


                        i += 1
                    maya_prs_original_attr = None
                    if type(temp_maya_nodes_curves)  == dict:
                        current_node_from_attr = cmds.connectionInfo("%s.output" % each_curve, \
                                                                        destinationFromSource=True)
                        source_node_attr = current_node_from_attr[0]
                        attribute_value_here = source_node_attr.split(".")[-1]
                        if source_node_attr in temp_maya_nodes_curves.values():
                            for current_key, current_value in temp_maya_nodes_curves.items():
                                if current_value == source_node_attr:
                                    splitter_attribute = current_value.split(".")[1]
                                    if attribute_value_here.find(splitter_attribute) !=-1:
                                        split_current_key = current_key.split(".")
                                        maya_prs_node = split_current_key[0]
                                        maya_prs_original_attr =  split_current_key[1]
                                        maya_prs_strip_attr = maya_prs_original_attr[0:-1]
                                        del (temp_maya_nodes_curves[current_key])
                                        break
                                else:
                                    continue
                    else:
                        maya_prs_node = current_node_prs_name
                        maya_prs_strip_attr = node_attr_type.lower()
                        maya_prs_original_attr =  current_attr_name

                    if not maya_prs_original_attr:
                        continue

                    kipCurveObject.keys.append(key)
                    kipCurveObject.name.append(maya_prs_node)
                    kipCurveObject.parm.append(maya_prs_strip_attr)

                    kipCurveObject.parm_attr.append(maya_prs_original_attr)

                    each_curve_it = None
                    if curves_from_maya_node:
                        for each_curve_it in curves_from_maya_node:
                            if maya_prs_node in each_curve_it:
                                break
                            else:
                                each_curve_it = None

                    if each_curve_it:
                        get_append_curve_v = kipCurveObject.output()
                        for each_append_cruve in get_append_curve_v:
                            each_curve_it[2].append(each_append_cruve)
                    else:
                        kipCurveClassObject.name = maya_prs_node
                        eye_type = "default"
                        if each_node in left_eye_curves:
                            eye_type = "left"
                        elif each_node in right_eye_curves:
                            eye_type = "right"

                        kipCurveClassObject.type = eye_type
                        kipCurveClassObject.animation.append(kipCurveObject)
                        curves_from_maya_node.append(kipCurveClassObject.output())
                current_prser_index += 1
            except Exception:
                current_prser_index += 1
                continue


        rodin_logger.info("Finished maya exporting data")
        return (curves_from_maya_node)

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

