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


import math
#import nuke
from rodin import logging
from kip.kip_curve_class import *
from kip.utils.kipError import *

rodin_logger = logging.get_logger('kipNuke')


class NodeCurves():
    """
    Creating nuke curve node class

    *Parents:*

        None

    *Children:*

        * :func:`getCurves`

    """
    def __init__(self):
        """
        Base init function for nuke cuve create Class.
        """
        rodin_logger.info("kip nuke nuke class initialized")
        self.precision = 3
        self.nuke_knobs = None
        self.nuke_tan_types = {1:"flat", 3:"linear", 0:"spline", 18:"fixed", \
                                    4:"clamped", 16:"spline", 2:"spline"}

    def getCurves(self, nuke_nodes_curves = [], nuke_attribute_curves = [],
                                left_eye_curves = [], right_eye_curves = []):
        """
        This function will get animation curve from nuke nodes and all anim curve data will be processed and it will generate a curve class.In silent mode, unsupported curve types will be replaced with a default curve type of user.

        :param nuke_nodes: list of nuke objects

        :type nuke_nodes: list

        :param nuke_attribute_curves: which attribute want to be transferd

        :type nuke_attribute_curves: list

        :param left_eye_curves: Left eye objects

        :type left_eye_curves: list

        :param right_eye_curves: Right eye objects

        :type right_eye_curves: list

        :return: Curve class object

        :rtype: curve

        Example

            >>> curve_obj = getCurves(nuke_nodes = ["Sphere1","Sphere2"],nuke_attribute_curves=["uniform_scale"],left_eye_curves=["Sphere1"],right_eye_curves=["Sphere2"])

        """
        curve_objects = []
        nuke_node = None
        if not nuke_nodes_curves:
            raise KipBaseError("Exporter need object list to export data !"\
                                                " Nothing found to Export!")

        build_node_list = []
        build_dest_node_list = {}
        if type(nuke_nodes_curves)  == dict:
            for each_nd in nuke_nodes_curves.keys():
                node_name_tr = nuke_nodes_curves[each_nd].split(".")[0]
                dest_node_name_tr = each_nd.split(".")[0]
                build_node_list.append(node_name_tr)
                build_dest_node_list.update({node_name_tr:dest_node_name_tr})
        elif type(nuke_nodes_curves) == list:
            build_node_list = nuke_nodes_curves
        else:
            raise KipBaseError("Unknown mapping structure found ! list and dict only supported!")
        build_node_list = set(build_node_list)

        try:
            for each_node in build_node_list:
                kipCurveClassObject = ClassObject()
                kipCurveObject = CurveClass()
                nuke_node = nuke.toNode(each_node)
                if nuke_node:
                    if nuke_attribute_curves:
                        self.nuke_knobs = nuke_attribute_curves
                    else:
                        self.nuke_knobs = nuke_node.knobs()
                    for each_knob in self.nuke_knobs:
                        if nuke_node[each_knob].isAnimated():
                            node_anim_curves = nuke_node[each_knob].animations()
                            curve_count = len(node_anim_curves)
                            channel_mark = {0:"X", 1:"Y", 2:"Z"}
                            for i in range(0, curve_count):
                                if len(node_anim_curves[i].knob().array()) == 3:
                                    channel_node = node_anim_curves[i].knob().label()
                                    channel_attr = "%s%s" % (node_anim_curves[i].knob().\
                                                                label(), channel_mark[i])
                                else:
                                    channel_node = node_anim_curves[i].knobAndFieldName()
                                    channel_attr = channel_node
                                if nuke_attribute_curves:
                                    for each_pass_attr in nuke_attribute_curves:
                                        if channel_attr.find(each_pass_attr) == -1:
                                            continue

                                get_all_keys = node_anim_curves[i].keys()
                                key = CurveKeyClass()
                                for each_key in get_all_keys :
                                    key.time.append(each_key.x)
                                    key.value.append(each_key.y)
                                    key.in_weight.append(each_key.la)
                                    key.out_weight.append(each_key.ra)
                                    key.in_tan_type.append(self.nuke_tan_types\
                                                        [each_key.interpolation])
                                    key.out_tan_type.append(self.nuke_tan_types\
                                                        [each_key.extrapolation])
                                    rad_in_angle = math.degrees(each_key.la)
                                    rad_out_angle = math.degrees(each_key.ra)
                                    key.in_angle.append(rad_in_angle)
                                    key.out_angle.append(rad_out_angle)
                                    key.in_slope.append(math.tan(rad_in_angle))
                                    key.out_slope.append(math.tan(rad_out_angle))

                                nuke_prs_original_attr = channel_node
                                nuke_prs_strip_attr = channel_attr

                                if type(nuke_nodes_curves)  == dict:
                                    if channel_attr.find(".")!= -1:
                                        current_node_prs = channel_attr.split(".")[0]
                                    else:
                                        current_node_prs = channel_attr
                                    source_node_attr = "%s.%s" % (nuke_node.name(), \
                                                                    current_node_prs)

                                    if source_node_attr in nuke_nodes_curves.values():
                                        current_key = (key for key, value in \
                                            nuke_nodes_curves.items() if \
                                                value==source_node_attr).next()

                                        split_current_key = current_key.split(".")
                                        nuke_prs_original_attr =  split_current_key[1]
                                        nuke_prs_strip_attr = nuke_prs_original_attr
                                kipCurveObject.keys.append(key)
                                kipCurveObject.parm.append(nuke_prs_original_attr)
                                kipCurveObject.parm_attr.append(nuke_prs_strip_attr)

                    current_nk_node_name = nuke_node.name()
                    each_nk_node = nuke_node.name()
                    if current_nk_node_name in build_dest_node_list.keys():
                        each_nk_node = build_dest_node_list[current_nk_node_name]

                    each_curve_it = None
                    if curve_objects:
                        for each_curve_it in curve_objects:
                            if each_nk_node in each_curve_it:
                                break
                            else:
                                each_curve_it = None

                    if each_curve_it:
                        get_append_curve_v = kipCurveObject.output()
                        for each_append_cruve in get_append_curve_v:
                            each_curve_it[2].append(each_append_cruve)
                    else:
                        kipCurveClassObject.name = each_nk_node
                        eye_type = "default"
                        if each_node in left_eye_curves:
                            eye_type = "left"
                        elif each_node in right_eye_curves:
                            eye_type = "right"
                        kipCurveClassObject.type = eye_type

                        kipCurveClassObject.animation.append(kipCurveObject)
                        curve_objects.append(kipCurveClassObject.output())
            rodin_logger.info("Finished getting nuke knob %s" % channel_node)
            return curve_objects
        except Exception:
            raise KipBaseError("Unable to get knob values from %s " % nuke_node)

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

