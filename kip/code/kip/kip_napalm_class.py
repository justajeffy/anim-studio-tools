#                 Dr. D Studios - Software Disclaimer
#
# Copyright 2009 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#
###############################################################################

"""
This module will help to write out anim curves as a napalm file or a xml file.
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
import traceback
import time
import napalm.core as nap_core
from rodin import logging
from rodin.decorators import deprecated

rodin_logger = logging.get_logger('kip')

class Napalm():
    """
    Main napalm class for writing napalm or xml file , this module will output napalm table
    structure with subtables based on the data recevied from curve class object.If writer
    files to write napalm file then it will return False .

    *Parents:*

        None

    *Children:*

        * :func:`writeNapalm`

        * :func:`writeMappingTable`

        * :func:`write`

        * :func:`read`

        * :func:`getAtter`

        * :func:`getChannel`

        * :func:`getAllChannles`

    """

    def __init__(self):
        """

        Base init function for Napalm Class.

        """
        self.napalm_data = []
        self.mapping_data = []

    def writeNapalm(self, nap_file_name, curve_object, debug = False,
                            map_file_name = None, software = None, app_version = None):
        """

        This function will write curve object to napalm file and napalm channel file.

        :param nap_file_name: Filepath for writting napalm data

        :type nap_file_name: string

        :param curve_object: Curve object to convert as napalm file

        :type curve_object: curve class object

        :param debug: This option will turn on the debug output

        :type debug: bool

        :param map_file_name: Filepath for writting napalm channel data

        :type map_file_name: string

        .. note::

            This function will support only few tangents and list is

            * flat

            * linear

            * spline

            * fixed

            * clamped

        .. warning::

            This function will endup with error if you pass wrong curve object structure.

        .. seealso::

            * :func:`writeMappingTable`

        .. versionchanged:: 0.0.5

            Fixed the map_file_name fixed.

        .. todo::

            More in-line comment should be added

        :return: Write Status,Nap File Path,Map File Path

        :rtype: bool,string,string

        Example

            >>> <ObjectTable @ 0x241f730>
                0:   <FloatBuffer at 0x23826e0 (FloatCpuStore[4] at 0x2417190)>
                1:   <StringBuffer at 0x24a48f0 (StringCpuStore[4] at 0x239c330)>
                2:   <FloatBuffer at 0x2411040 (FloatCpuStore[4] at 0x24bed30)>
                3:   <StringBuffer at 0x2410c80 (StringCpuStore[4] at 0x2410cc0)>
                4:   <FloatBuffer at 0x239a630 (FloatCpuStore[4] at 0x239a670)>
                5:   <FloatBuffer at 0x2398120 (FloatCpuStore[4] at 0x2398160)>
                6:   <FloatBuffer at 0x239bb80 (FloatCpuStore[4] at 0x239bbc0)>
                7:   <FloatBuffer at 0x2388c20 (FloatCpuStore[4] at 0x2388c60)>
                8:   <FloatBuffer at 0x2388e90 (FloatCpuStore[4] at 0x238af50)>
                9:   <FloatBuffer at 0x238b150 (FloatCpuStore[4] at 0x238b190)>
                10:  <FloatBuffer at 0x22405c0 (FloatCpuStore[4] at 0x2240600)>

        """

        self.napalm_data = curve_object
        nap_main_table  = nap_core.ObjectTable()
        nap_status = False
        nap_file = None
        map_file = None
        counter_index = 0
        for each_node in self.napalm_data:
            curve_class = each_node[2]
            object_node = []
            object_dict = {}
            for eachCurve in curve_class:
                flot_attr_list = ['time', 'key_value', 'in_weight', 'out_weight', 'in_angle', \
                                                        'out_angle', 'in_slope', 'out_slope']
                curve_attr = str(eachCurve[1])
                map_data = {}
                for key in eachCurve[-1].keys():
                    dict_key_val = eachCurve[-1][key]
                    if key in flot_attr_list:
                        nap_main_table[counter_index] = nap_core.FloatBuffer(len(dict_key_val))
                    else:
                        nap_main_table[counter_index] = nap_core.StringBuffer(len(dict_key_val))
                    nap_main_table[counter_index].contents = dict_key_val
                    map_data.update({key:counter_index})
                    counter_index += 1
                object_node.append([curve_attr, map_data])
            object_dict.update({each_node[0]:[each_node[1], object_node]})
            self.mapping_data.append(object_dict)

        if debug:
            nap_core.dump(nap_main_table)
        try:
            nap_core.save(nap_main_table, nap_file_name)
            map_file = self.writeMappingTable(nap_file_name, map_file_name, \
                                                software = software, app_version = app_version)
            nap_file = nap_file_name
            nap_status = True
            self.napalm_data = []
            self.mapping_data = []
        except:
            traceback.print_exc()
            nap_status = False

        return (nap_status, map_file, nap_file)

    def writeMappingTable(self, nap_file_name, map_file_name = None, software = None,
                                                                    app_version = None):

        """

        This function will write curve object to napalm file and napalm channel file.

        :param nap_file_name: Filepath for writting napalm data'

        :type nap_file_name: string

        :param map_file_name: Filepath for writting napalm channel data

        :type map_file_name: string

        .. note::

            This function will write out mapping data for the channel.

        Example

            >>> "pSphere2":  <ObjectTable @ 0x1cfe910>
                            "eye_val":     "right"
                            "rotateX":     <ObjectTable @ 0x1cfea90>
                            "in_angle":      99
                            "in_slope":      98
                            "in_tan_type":   93
                            "in_weight":     92
                            "key_value":     90
                            "out_angle":     94
                            "out_slope":     96
                            "out_tan_type":  91
                            "out_weight":    95
                            "time":          97

        .. seealso::

            * :func:`writeNapalm`

        .. versionchanged:: 0.0.5

            Fixed the map_file_name fixed.

        .. todo::

            More in-line comment should be added

        :return: Map File Path

        :rtype: string

        """

        if not map_file_name:
            ext_spliter = os.path.splitext(os.path.basename(nap_file_name))
            get_file_ext = ext_spliter[-1]
            set_file_base = "%s_map%s" % (ext_spliter[0], get_file_ext)
            map_file_name = "%s/%s" % (os.path.dirname(nap_file_name), set_file_base)

        map_main_table  = nap_core.ObjectTable()
        for each_map_obj in self.mapping_data :
            object_keys = each_map_obj.keys()
            map_obj_table  = nap_core.ObjectTable()
            for each_key in object_keys:
                obj_key_val = each_map_obj[each_key][-1]
                eye_value =  each_map_obj[each_key][-0]
                map_obj_table["eye_val"] = eye_value
                for each_curve in obj_key_val:
                    nap_map_table = nap_core.ObjectTable()
                    dic_val = each_curve[-1].keys()
                    for each_dict_key in dic_val:
                        nap_map_table[each_dict_key] = each_curve[-1][each_dict_key]
                    map_obj_table[each_curve[0]] = nap_map_table

            map_main_table[each_key] = map_obj_table

        header_table = nap_core.ObjectTable()
        software_arg = str(software).split(",")[0]
        version = str(software).split(",")[1]
        date_time = time.strftime("%m/%d/%y-%H-%M-%S", time.localtime())
        header_table["nap_file"] = nap_file_name
        header_table["date_time"] = date_time
        header_table["kip_version"] = os.getenv("DRD_KIP_VERSION")
        header_table["app_version"] = app_version
        header_table["client_software"] = software_arg
        header_table["client_version"] = version
        header_table["user"] = os.getenv("USER")

        map_main_table["header"] = header_table

        try:
            nap_core.save(map_main_table, map_file_name)
            map_file_name = map_file_name
        except:
            traceback.print_exc()
            map_file_name = None

        return map_file_name

    def write(self, nap_file_name, curve_object, debug = False, map_file_name = None,
                                                    software = None, app_version = None):

        """

        This function will write curve object to napalm file and napalm channel file.

        :param nap_file_name: Filepath for writting napalm data

        :type nap_file_name: string

        :param map_file_name: Filepath for writting napalm channel data

        :type map_file_name: string

        :param curve_object: Curve object to convert as napalm file

        :type curve_object: curve class object

        :param debug: This option will turn on the debug output

        :type debug: bool

        .. note::

            This function will write out mapping data for the channel and napalm curve file..

        Example

            >>> import kip.kip_napalm_class as knc

            >>> kncw = knc.Napalm()

            >>> stat,n_file,m_file = kncw.write("/tmp/maya_kip_test_s.nap",curve_object,\
                                    debug=True,map_file_name="/tmp/maya_kip_test_m.nap")

            >>> True,/tmp/maya_kip_test_s.nap,/tmp/maya_kip_test_m.nap

        .. seealso::

            * :func:`writeNapalm`
            * :func:`writeMappingTable`

        .. versionchanged:: 0.0.5

            Fixed the map_file_name fixed.

        .. todo::

            More in-line comment should be added

        :return: Write Status,Nap File Path,Map File Path

        :rtype: bool,string,string

        """

        get_stat, map_file, nap_file = self.writeNapalm(nap_file_name, curve_object,
                                debug = debug, map_file_name = map_file_name, \
                                software = software, app_version = app_version)
        return (get_stat, map_file, nap_file)

    def read(self, filename):
        """

        This function will return napalm table data from napalm file.

        :param nap_file_name: Filepath to read napalm file.

        :type nap_file_name: string

        :return: Napalm Table Structure

        :rtype: ObjectTable

        """
        if filename:
            nap_table = nap_core.load(filename)
            return nap_table

    @deprecated(message="This function is removed use getChannel or getAllChannles ")
    def getAtter(self, node, curve_table, node_check=False):
        """
        This function will retrun channel value from the curve table
        """
        nap_node = curve_table["node"]

        if nap_node != node:
            if not node_check:
                return nap_node["sub_table"]
            else:
                return None
        else:
            return nap_node["sub_table"]


    def getChannel(self, nap_file_name, channel_number):
        """

        This function will return channel data based on the channel number

        :param nap_file_name: Filepath to read napalm file.

        :type nap_file_name: string

        :param channel_number: Channel number for getting data from the channel file.

        :type channel_number: int

        :return: Napalm Object

        :rtype: NapalmBuffer

        Example

            >>> import kip.kip_napalm_class as knc

            >>> kncw = knc.Napalm()

            >>> nap_tab = kncw.getChannel(nap_file_name,50)

        """
        if os.path.exists(nap_file_name):
            chan_table = nap_core.load(nap_file_name)
            get_channel_value = chan_table[channel_number].contents
            return get_channel_value

    def getAllChannles(self, nap_file_name):
        """

        This function will return all channel data from a channel file.

        :param nap_file_name: Filepath to read napalm file.

        :type nap_file_name: string

        :return: Napalm Object

        :rtype: NapalmBuffer

        Example

            >>> import kip.kip_napalm_class as knc

            >>> kncw = knc.Napalm()

            >>> nap_tab = kncw.getAllChannles(nap_file_name)

            >>> nap_tab["key_value"].contents

        """

        if os.path.exists(nap_file_name):
            chan_table = nap_core.load(nap_file_name)
            return(chan_table)



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

