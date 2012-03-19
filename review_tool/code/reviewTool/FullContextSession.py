import      os, re, traceback, time

from        drTank.util         import shotgun_session

import      projectAwareness
from        projectAwareness    import get_tank_render_container

from        util                import is_valid_render
import      tank

import ContextSession
from ContextSession import get_tank_obj_from_rv_link

from reviewTool import ASSET_RENDER_TYPES

class Shot(ContextSession.Shot):
    def __init__(self, scene, shot, project=None, shotgun_fields={}):

        ContextSession.Shot.__init__(self, scene, shot, project, shotgun_fields)

        if scene.name in ASSET_RENDER_TYPES:
            self.name = shot

    def get_all_render_from_shotgun(self, department=None, cachedOnly = False):
        return self.scene.get_all_shot_render_fast(self.name, department, cachedOnly=cachedOnly)

    def get_all_render(self, department=None, render_type = "Frames"):
        # get the lensing data
        container_address = get_tank_render_container(  project_name = self.project.name,
                                                        scene_name = self.scene.name,
                                                        shot_name = self.name,
                                                        department = department
                                                        )

        # later should also get the revision type dynamically
        try:
            container = tank.local.find(container_address)
        except tank.common.TankNotFound:
            return []

        rev_list = container.get_revisions(tank.local.find(render_type))

        # do some checks to ensure it's a valid publish
        rev_list = [ r for r in rev_list
                            if is_valid_render(r.get_filesystem_location(), render_type, check_filesize=True)
                    ]

        rev_list.sort(lambda x,y: cmp( x.get_creation_date(), y.get_creation_date() ) , reverse=True)

        return rev_list


ORDER_ID_ZERO_PADDING = 6   # the number of zero to pad scene or shot that doesn't have a order
class Scene(ContextSession.Scene):

    def __init__(self, scene, project=None, shotgun_fields={}):
        ContextSession.Scene.__init__(self,scene, project, shotgun_fields )

        # for storing character, props, env, etc, renders
        self.flg_has_cache_shot_render = False
        self._cache_asset_render = {}

    def clear_shotgun_render_cache(self):
        self._cache_asset_render = {}
        self.flg_has_cache_shot_render = False
        self._cache_shot_list   = None
        self._cache_shot_render = None

    def get_all_shot_render_fast(self, shot_name, department=None,cachedOnly=False):
        # handle the shots render request
        if not self.name in ASSET_RENDER_TYPES:
            return ContextSession.Scene.get_all_shot_render_fast(self, shot_name, department, cachedOnly = cachedOnly )

        else:
            self.get_all_scene_render_shotgun()
            return self._cache_asset_render[self.name][department][shot_name]



    def get_all_scene_render_shotgun(self):
        if not self.name in ASSET_RENDER_TYPES:
            return ContextSession.Scene.get_all_scene_render_shotgun(self)

        else:
            if not self._cache_asset_render.has_key(self.name):
                self._cache_asset_render[self.name] =  self._get_render_data_shotgun(search_spec =
                                                     [['sg_tank_address','contains','%s(' % self.name],
                                                      ["sg_status_list","is_not", "omt"],
                                                      ])

            return self._cache_asset_render[self.name]


    def list_shots(self):
        '''
        List the scene in shotgun scene order
        '''
        if self._cache_shot_list==None:
            if self.name in ASSET_RENDER_TYPES:
                if self.name=="Character":
                    char_list = shotgun_session().list_character()
                    char_var_list = shotgun_session().list_character_var()
                    asset_list = char_list + char_var_list
                else:
                    asset_list = {
                                    'Prop':shotgun_session().list_prop,
                                    'Stage':shotgun_session().list_stage,
                                    'Environment':shotgun_session().list_environment,
                                    'Skydome':shotgun_session().list_skydome,
                                  }[self.name]()


                asset_list.sort(lambda x, y: cmp(x['code'], y['code']))
                self._cache_shot_list = []

                i=0
                for asset_data in asset_list:
                    asset_data["sg_cut_order"] = i
                    asset_data["sg_handle_start"] = 0
                    asset_data["sg_handle_end"] = 0
                    asset_data["sg_cut_start"] = 0
                    asset_data["sg_cut_end"] = 0

                    i+=1
                    self._cache_shot_list.append( Shot(
                                                        shot              = asset_data["code"],
                                                        scene             = self,
                                                        project           = self.project,
                                                        shotgun_fields    = asset_data,
                                                       ))



            else:
                ContextSession.Scene.list_shots(self, Shot)

        return self._cache_shot_list

#    def generate_shot_order(self, shot_name):
#        if "_" in shot_name: shot_name = shot_name.split("_")[1]
#
#        shot_name_int = int( re.search("([0-9]+)",  shot_name ).groups()[0] )
#
#        return pow(10, ORDER_ID_ZERO_PADDING) + shot_name_int
#
#    def get_neighbour_shot(self, shot):
#        shot_name = shot.name if isinstance(shot, Shot) else shot
#        scene_name = self.name
#
#        # search for the scene with the matchin name in the scene cache
#        shots = self.list_shots()
#        shot_obj_search = [ shot for shot in shots if shot.name == shot_name ]
#
#        shot = shot_obj_search[0] if shot_obj_search else None
#
#        if not shot: raise KeyError, "Scene '%(scene_name)s', Shot '%(shot_name)s' is not found in shotgun." % vars()
#
#        # now find the neighbours
#        index = shots.index(shot)
#
#        prev = shots[index-1] if index!=0 else None
#        next = shots[index+1] if index!=len(shots)-1 else None
#
#        return prev, next
#
#    def neighbours(self):
#        return self.project.get_neighbour_scenes(self)
#
#
#    def __repr__(self):
#        return "Scene object: %s" % self.name


#def get_tank_obj_from_rv_link(rv_link_string):
#    # get the tank obj from the movie link
#    result = re.search("\.(?P<entity_id>[0-9]+)_(?P<revision_id>[0-9]+)\.mov", rv_link_string)
#
#    if result: # ie movie is not viewable through shotgun, hence don't show in review tool either
#
#        tank_obj = tank.local.Em().get_revision_by_id(int(result.groupdict()["entity_id"]),
#                                                      int(result.groupdict()["revision_id"]))
#        return tank_obj


global _global_project
_global_project = None

def get_project(project=None):
    global _global_project
    if not _global_project:
        _global_project = Project(project, singletonOnly=True)

    return _global_project



class Project(ContextSession.Project):
    def __init__(self, project, singletonOnly):
        ContextSession.Project.__init__(self, project, singletonOnly)

    def clear_shotgun_render_cache(self):
        for scene in self.list_scenes():
            scene.clear_shotgun_render_cache()

#    def get_scene(self, scene):
#        # get the scene name from parameter, which can either be a string, or a scene object
#        scene_name = scene.name if isinstance(scene, Scene) else scene
#
#        # search for the scene with the matchin name in the scene cache
#        scenes = self.list_scenes()
#        scene_obj_search = [ scene for scene in scenes if scene.name == scene_name ]
#
#        scene = scene_obj_search[0] if scene_obj_search else None
#
#        return scene


    def list_scenes(self):
        '''
        List the scene in shotgun scene order
        '''
        if self._cache_scene_list == None:
            ContextSession.Project.list_scenes(self, Scene)

            for asset_type in ASSET_RENDER_TYPES:
                self._cache_scene_list.insert(ASSET_RENDER_TYPES.index(asset_type),

                                              Scene( scene            = asset_type,
                                                     project          = self,
                                                     shotgun_fields   = {'id':asset_type,
                                                                        'code':asset_type,
                                                                        'sg_sort_order':-100+ASSET_RENDER_TYPES.index(asset_type)}))

        return self._cache_scene_list

#
#
#    def list_scene_names(self):
#        return [s.name for s in self.list_scenes()]

#    def get_neighbour_scenes(self, scene):
#        # get the scene name from parameter, which can either be a string, or a scene object
#        scene_name = scene.name if isinstance(scene, Scene) else scene
#
#        # search for the scene with the matchin name in the scene cache
#        scenes = self.list_scenes()
#        scene_obj_search = [ scene for scene in scenes if scene.name == scene_name ]
#
#        scene = scene_obj_search[0] if scene_obj_search else None
#        if not scene: raise KeyError, "Scene with name '%s' is not found in shotgun." % scene_name
#
#        # now find the neighbours
#        index = scenes.index(scene)
#
#        prev = scenes[index-1] if index!=0 else None
#        next = scenes[index+1] if index!=len(scenes)-1 else None
#
#        return prev, next

#
#    def __repr__(self):
#        return "Project object: %s" % self.name
#
#
#    @staticmethod
#    def _construct(project):
#        """ create the project context object """
#        return project if isinstance(project, Project) else Project(project)


#def _sort_shotgun_clips(y, x):
#    result = cmp(x["date"], y["date"])
#    if result ==0:
#        return cmp(x['name'], y['name'])
#    else:
#        return result

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

