import os
import tank
import drTank.util

import re

SHOT_RENDER_CONTAINER   = os.environ.get("SHOT_RENDER_CONTAINER", "ShotRender_v3")
SHOT_AUDIO_CONTAINER    =  os.environ.get("SHOT_AUDIO_CONTAINER",  "ShotAudio_v2")
ASSET_RENDER_CONTAINER  = {
                          "Character":  "CharacterRender_v2",
                          "Prop":       "PropRender_v2",
                          "Stage":      "StagRender_v2",
                          "Environment":"EnvironmentRender_v2",
                          "Skydome":    "SkydomeRender_v2",
                          }

from reviewTool import ASSET_RENDER_TYPES

def format_shotgun_shot_name(project_name, scene_name, shot_name):
    if project_name=="hf2" and scene_name not in ASSET_RENDER_TYPES and not(shot_name.startswith(scene_name+"_")):
        return "%(scene_name)s_%(shot_name)s" % vars()

    return shot_name

def get_tank_shot_render_container_name(project_name):
    if project_name == "hf2":
        return SHOT_RENDER_CONTAINER

def get_tank_render_container(project_name, scene_name, shot_name, department):
    from reviewTool import ASSET_RENDER_TYPES

    if project_name == "hf2":
        shot_render_container = SHOT_RENDER_CONTAINER
        shot_name = format_shotgun_shot_name(project_name, scene_name, shot_name)

        if department != "previs" and shot_render_container.endswith("v2"):
            return "%(shot_render_container)s(Scene(%(scene_name)s), Department(%(department)s), SceneShot(%(shot_name)s))" % vars()

        elif scene_name in ASSET_RENDER_TYPES:
            container = ASSET_RENDER_CONTAINER[scene_name]
            return "%(container)s(ReviewType(creative), Department(%(department)s), %(scene_name)s(%(shot_name)s))" % vars()

        else:
            return "%(shot_render_container)s(Scene(%(scene_name)s), Department(%(department)s), SceneShot(%(shot_name)s), ReviewType(creative))" % vars()


def get_tank_audio_container(project_name, scene_name, shot_name, audio_name="master"):
    if project_name == "hf2":
        shot_audio_container = SHOT_AUDIO_CONTAINER

        return "%(shot_audio_container)s(%(audio_name)s, Scene(%(scene_name)s), SceneShot(%(shot_name)s))" % vars()


def is_tank_render_container(project_name, obj):
    import tank
    if project_name == "hf2":
        if "Render_v" in obj.get_entity_type().get_name(): #in [SHOT_RENDER_CONTAINER, "CharacterRender"]:
            return True

def get_scene_shot_from_tank_address(address):

    asset = tank.find(address).asset

    return asset.labels["Scene"].system.name, asset.labels["SceneShot"].system.name


def get_shot_list_from_scene(scene_name):
    shot_list = []
    from drTank.util import shotgun_session as ss
    api = ss().api()
    shot_dicts = api.find_one("Scene", [["code", "is", scene_name]], ["shots"])["shots"]
    for dict in shot_dicts:
        shot_list.append(dict["name"])
    return shot_list





def determine_aux_data_from_tank_address_fast(tank_address, include_shotgun_data=False):
    try:
        container_name = ""
        if "ShotRender" in tank_address:
            rev_name    = re.search("(Frames|Movie)\(([\w]+),", tank_address).groups()[1]
            shot        = re.search("Shot\(([\w]+)\)", tank_address).groups()[0]
            scene       = re.search("Scene\(([\w]+)\)", tank_address).groups()[0]

        elif "Render" in tank_address:
            asset_type  = re.search("([\w]+)Render_v[0-9]{1,2}", tank_address)
            rev_name    = re.search("(Frames|Movie)\(([\w-]+),", tank_address)
            if rev_name:
                rev_name= rev_name.groups()[1]
            else:
                return {}

            # for assets, the shot is the name of the asset
            if asset_type!=None:
                asset_type = asset_type.groups()[0]

                container_name = re.search(asset_type+"Render_v[0-9]{1,2}\(([\w]+)", tank_address).groups()[0]
                shot        = re.search(asset_type+"\(([\w]+)\)", tank_address).groups()[0]
                scene       = asset_type

                if container_name!="default":
                    shot = "%s%s%s" % (shot, container_name[0].upper(), container_name[1:] )

            else: # general render type, ie "Movie(001, Render_v1(default, Department(surface)))"
                scene       = "General"
                shot        = "default"  # to be updated

        review_type = re.search("ReviewType\(([\w]+)\)", tank_address)
        review_type = review_type.groups()[0] if review_type else "creative"
        department  = re.search("Department\(([\w]+)\)", tank_address)
        rev_name    = rev_name if (review_type.lower()=="creative") else ("%s T" % rev_name)

        data_hash = {   "rev_name": rev_name,
                        "container_name":container_name,
                        "scene":scene,
                        "shot":shot,
                        "context_type":"shot",
                        "department":department.groups()[0] if department else "",
                        "review_type": review_type # default to creative
                        }


        data_hash["cut_range"] = ""
        data_hash["cut_order"] = 0
        data_hash["cut_order_index"] = 0

        if include_shotgun_data:
            import ContextSession
            c_shot = ContextSession.get_project().get_scene(scene).get_shot(shot)
            data_hash["cut_range"] = c_shot.get_cut_info()
            data_hash["cut_order"] = c_shot.get_cut_order(as_string=True)
            data_hash["cut_order_index"]    = c_shot.get_cut_order()

        return data_hash

    except:
        print "Warning: can not determine aux data for address %s fast." % tank_address
        import traceback
        traceback.print_exc()

        return {}


def determine_aux_data_from_tank_address(tank_rev, include_shotgun_data=False):
    import tank
    if type(tank_rev) in (str,unicode):
        tank_rev = tank.find(tank_rev)

    return  determine_aux_data_from_vfs_path(tank_rev = tank_rev, include_shotgun_data=include_shotgun_data )


def get_project():
    return os.environ.get("DRD_JOB")

def determine_aux_data_from_vfs_path(path=None, tank_rev=None, project=None, include_shotgun_data=False):
    '''
    Given the tank vfs path, return
    ex: {"context_type":"shot", "scene":"19d","shot":"19d_010"}

    include shotgun data will query additional shotgun information
    '''
    TANK_VFS_ROOT = tank.util.config.virtual_tree.fuse_mount_point
    import FullContextSession

    if project==None:
        project = os.environ.get("DRD_JOB", "hf2")

    if not tank_rev:
        if path and path.startswith(TANK_VFS_ROOT):

            try:
                if os.path.isfile(path):
                    tank_rev = tank.find(path)
                elif re.search("\.([0-9]+)_([0-9]+)\.[a-zA-Z0-9]+$", path):
                    tank_rev = tank.find(path)
                else:
                    tank_rev = tank.find(os.path.split(path)[0])

            except:
                print "Warning: failed to find tank object from path %s" % path
                import traceback
                traceback.print_exc()

        elif re.search("/([0-9]+)/r([0-9]+)_", path):
            useless, rid = re.search("/([0-9]+)/r([0-9]+)_", path).groups()
            tank_rev = tank.server.Em().get_revision_by_id(int(rid))
            tank_rev = tank.find(str(tank_rev))

    if type(tank_rev)==tank.asset.tank_revision.TankRevision:
        tank_asset_rev = tank_rev
        tank_rev = tank_rev.get_object()

    # ensure the revision is from a tank render container

    if not tank_rev or not is_tank_render_container(project, tank_rev.get_asset()):
        return None

    # now we have the tank address, get the scene shot
    data_hash = determine_aux_data_from_tank_address_fast(str(tank_rev))

    if data_hash.has_key("scene") and data_hash["scene"] and data_hash["shot"]:
        data_hash.update( {
                        #"rev_name": tank_rev.get_name(),
                        "context_type":"shot",
                        'tank_asset_rev_obj':tank_asset_rev,
                        'tank_rev_obj':tank_rev,
                        'tank_rev_id':(tank_rev.get_asset().get_id(), tank_rev.get_id()),
                        })

        if include_shotgun_data and FullContextSession.get_project().get_scene(data_hash["scene"]):
            c_shot = FullContextSession.get_project().get_scene(data_hash["scene"]).get_shot(data_hash["shot"], no_exception=True)
            if c_shot:
                data_hash["cut_range"] = c_shot.get_cut_info()
                data_hash["cut_order"] = c_shot.get_cut_order(as_string=True)
                data_hash["cut_order_index"]    = c_shot.get_cut_order()

        return data_hash
    else:
        print "Warning: can not determine scene shot from address %s..." % tank_rev
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

