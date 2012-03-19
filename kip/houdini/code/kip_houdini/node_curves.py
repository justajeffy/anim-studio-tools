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
import math
from rodin import logging
from kip.kip_curve_class import *
from kip.utils.kipError import *

rodin_logger = logging.get_logger('kipHoudini')
GLOBAL_FPS = 24
GLOBAL_TIME = 1

class NodeCurves():
    """
    Creating houdini curve node class

    *Parents:*
        None

    *Children:*

        * :func:`getCurves`

        * :func:`animCurvesFromNode`

    """
    def __init__(self):
        """
        Base init function for houdini cuve create Class.
        """
        self.supported_tan_types = ["spline", "linear", "bezier", "constant", "cubic"]
        self.channel_match = {'tx':'translateX', 'ty':'translateY', 'tz':'translateZ', \
                                'rx':'rotateX', 'ry':'rotateY', 'rz':'rotateZ', \
                                'sx':'scaleX', 'sy':'scaleY', 'sz':'scaleZ'}

        rodin_logger.info("kip houdini curve class initialized")

    def getCurves(self, houdini_node_curves = [], houdini_attribute_curves = [], \
                        silent = False, start_frame = None, end_frame = None, \
                        left_eye_curves = [], right_eye_curves = []):
        """
        This function will get animation curve from houdini nodes and all anim curve data will be processed and it will generate a curve class

        :param houdini_nodes: list of houdini objects

        :type houdini_nodes: list

        :param silent: If this option is true then we will ignore all tangent errors.

        :type silent: bool

        :param start_frame: start frame to capture

        :type start_frame: int

        :param end_frame: end frame to capture

        :type end_frame: int

        :return: Curve class object

        :rtype: curve

        Example

            >>> curve_obj = getCurves(houdini_nodes = ["/obj/geo/xform_1","/obj/geo/xform_2"],start_frame = 10,end_frame = 100,silent = False,left_eye_curves=["/obj/geo/xform_1"],right_eye_curves=["/obj/geo/xform_2"])

        """
        curves_from_houdini_node = []

        if not houdini_node_curves:
            raise KipBaseError("Exporter need object list to export data ! "\
                                                "Nothing found to Export!")

        try:
            for each_node in houdini_node_curves:
                each_node = hou.node(each_node)
                kipCurveClassObject = ClassObject()
                kipCurveObject = CurveClass()
                anim_curves = self.animCurvesFromNode(each_node.path())
                if anim_curves:
                    for each_curve in anim_curves:
                        current_attr = each_curve.keys()[0]
                        if houdini_attribute_curves:
                            for each_pass_attr in houdini_attribute_curves:
                                if current_attr.find(each_pass_attr) == -1:
                                    continue

                        key = CurveKeyClass()
                        for each_time_curve in (each_curve[current_attr]):
                            key.time.append(round((each_time_curve.time()*GLOBAL_FPS)+GLOBAL_TIME))
                            key.value.append(each_time_curve.value())
                            try:
                                in_angle_v = each_time_curve.inSlope()
                            except hou.KeyframeValueNotSet:
                                in_angle_v = 0
                            try:
                                out_angle_v = each_time_curve.slope()
                            except hou.KeyframeValueNotSet:
                                out_angle_v = 0
                            key.in_angle.append(in_angle_v)
                            key.out_angle.append(out_angle_v)

                            try:
                                in_weight_v = each_time_curve.inAccel()
                            except hou.KeyframeValueNotSet:
                                in_weight_v = 0
                            try:
                                out_weight_v = each_time_curve.accel()
                            except hou.KeyframeValueNotSet:
                                out_weight_v = 0

                            key.in_weight.append(in_weight_v)
                            key.out_weight.append(out_weight_v)


                            in_tan_type_prs = str(each_time_curve.expression()[:-2])
                            if not in_tan_type_prs in self.supported_tan_types:
                                if silent:
                                    in_tan_type_prs = "user"
                                else:
                                    raise KipAttrubiteError("%s not supported tangent to nuke,"\
                                    " Please update anim curve with any one of this %s" % \
                                                (in_tan_type_prs, self.supported_tan_types))

                            key.in_tan_type.append(in_tan_type_prs)
                            key.out_tan_type.append(in_tan_type_prs)

                            rad_in_angle = math.radians(in_angle_v)
                            rad_out_angle = math.radians(out_angle_v)
                            key.in_slope.append(math.tan(rad_in_angle))
                            key.out_slope.append(math.tan(rad_out_angle))

                        kipCurveObject.keys.append(key)
                        current_node = hou.node(each_node.path())
                        current_parm = current_node.parm(current_attr)
                        kipCurveObject.parm.append(current_parm.parmTemplate().label().lower())
                        current_attr_pass = current_attr
                        if self.channel_match.has_key(current_attr):
                            current_attr_pass = self.channel_match[current_attr]
                        kipCurveObject.parm_attr.append(current_attr_pass)

                    kipCurveClassObject.name = each_node.name()
                    eye_type = "default"
                    if each_node.name() in left_eye_curves:
                        eye_type = "left"
                    elif each_node.name() in right_eye_curves:
                        eye_type = "right"

                    kipCurveClassObject.type = eye_type
                    kipCurveClassObject.animation.append(kipCurveObject)
                    curves_from_houdini_node.append(kipCurveClassObject.output())
                else:
                    rodin_logger.info("No animation curve found on %s "%each_node)
        except Exception:
            traceback.print_exc()
            raise KipBaseError("Error occured while writing anim curves from %s ."\
                    "Please check script editor for more error details" % each_node)

        rodin_logger.info("Finished houdini exporting data")
        return (curves_from_houdini_node)



    def animCurvesFromNode(self, node_path):
        """
        This is simple function for grabing all animation curve from a houdini node.

        .. note::

            This function is checking data is isTimeDependent command so if any kind of curve attached to a node with frame data it will treate as a animation curve.

        :param node_path: houdini node path to check animation values

        :type node_path: string

        Example

            >>> anim_curves = animCurvesFromNode("/obj/geo/xform_1")

        """
        node = hou.node(node_path)
        all_anim_curves = []
        if node:
            all_parms = node.parms()
            for each_prm in all_parms:
                if each_prm.isTimeDependent():
                    if len(each_prm.keyframes()) > 0:
                        curve_dict = {}
                        if not curve_dict.has_key(each_prm.name()):
                            curve_dict[each_prm.name()]=each_prm.keyframes()
                        else:
                            curve_dict[each_prm.name()]=each_prm.keyframes()
                        all_anim_curves.append(curve_dict)
        else:
            msg = "%s not exists in scene !" % node.path()
            rodin_logger.error(msg)
            raise KipBaseError(msg)
        return all_anim_curves

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

