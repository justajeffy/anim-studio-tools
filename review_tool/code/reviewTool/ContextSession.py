import      os, re, traceback, time

from        drTank.util         import shotgun_session

import      projectAwareness
from        projectAwareness    import get_tank_render_container

from        util                import is_valid_render
import      tank

# global fields used for sorting results
SORT_STATUS_ORDER   = {}                # populated by ContextWidget and used in the _sort_shotgun_clips method
SORT_METHOD         = 'sort_by_date'    # defines sorting mechanism (manipulated in the ContextWidget)

class Shot:
    def __init__(self, scene, shot, project=None, shotgun_fields={}):
        self.project    = Project._construct( project )
        self.scene      = Scene._construct( scene = scene, project = self.project )
        self.name       = projectAwareness.format_shotgun_shot_name(
                                                                    shot_name = shot,
                                                                    project_name = self.project.name,
                                                                    scene_name = self.scene.name )
        self._shotgun_fields = shotgun_fields
        self._cache_latest_tank_audio = None

    def neighbours(self):
        """
        Return the neighbour shots according to shotgun order

        """
        return self.scene.get_neighbour_shot(self)

    def __repr__(self):
        return "Shot object: %s" % self.name

    def get_cut_info(self, flg_get_latest_if_no_approved=False):
        """
        Return the cut information for a shot.
        ex: {'sg_handle_end': 1266, 'sg_handle_start': 997, 'sg_cut_end': 1258, 'sg_cut_start': 1005}
        """
        # attempt to get the approved cut data, if possible
        fields = ["sg_handle_start", "sg_handle_end", "sg_cut_start","sg_cut_end"]
        cut_data = self.shotgun_field(fields)
        shotgun_shot_id = self.shotgun_field('id')

        # if this doesn't exist have to query the cut data base for it.
        api = shotgun_session().api()

        if flg_get_latest_if_no_approved and not cut_data["sg_handle_start"]:
            print 'approved cut data not found, getting in from latest cut.'
            cut_data = api.find_one(
                                     'Cut_sg_shots_Connection',
                                     [['shot','is',{'type':'Shot','id':shotgun_shot_id}]],
                                     fields,
                                     [{'field_name':'id','direction':'desc'}]
                                    )

        # must check if cut_data not None
        if cut_data == None:
            pass
        elif cut_data["sg_handle_start"]==None:
            cut_data = None
        else:
            # filter the result, shotgun usually return with id and type, along with the query fields
            field_values = [ cut_data[f] for f in fields ]
            cut_data = dict(zip (  fields, field_values ))

        return cut_data

    def get_render_shotgun(self, department, rev_name):
        all_scene_render = self.scene.get_all_scene_render_shotgun()
        ver = [rev for rev in all_scene_render[department][self.name] if rev["name"]==rev_name]

        if ver:
            return ver[0]

    def get_latest_render(self, department=None, render_type = "Frames", return_tank_object=True):
        """
        Return the latest render for a shot.
        """
        # get the lensing data
        import tank

        container_address = get_tank_render_container(  project_name = self.project.name,
                                                        scene_name = self.scene.name,
                                                        shot_name = self.name,
                                                        department = department
                                                        )

        # later should also get the revision type dynamically
        try:
            rev = tank.local.find("%(render_type)s(latest, %(container_address)s)" % vars() )
        except tank.common.errors.TankNotFound, e:
            import traceback
#            traceback.print_exc()
            return None
        except tank.common.errors.TankAddressParseError:
            return None

        if return_tank_object:
            return rev
        else:
            return rev.system.vfs_full_paths[0]

    def get_latest_master_audio(self, revision_type="Audio", return_tank_object=False):
        """
        Return the latest audio for a shot.
        By default it returns the file path to the audio.
        If return_tank_object=True, it will return the Tank revision object.
        """
        import traceback

        if self._cache_latest_tank_audio == "_search_but_not_found":
            return None

        audio_container = projectAwareness.get_tank_audio_container(self.project.name,
                                                   self.scene.name,
                                                   self.name,
                                                   audio_name="master")

        try:
            audio_address = "%(revision_type)s(latest, %(audio_container)s)" % vars()
            rev = tank.find(audio_address)


        except tank.common.errors.TankNotFound, e:
            self._cache_latest_tank_audio = "_search_but_not_found"
            print "Tank Not Found: can not find audio under %s" % audio_address

            return None

        except tank.common.errors.TankAddressParseError:
            print "Tank Address Parse Error: can not find audio under %s" % audio_address

            return None

        if return_tank_object:
            return rev
        else:
            return rev.system.vfs_full_paths[0]


    def get_tank_render(self, department, version, revision_type="Frames"):
        import tank

        render_container = projectAwareness.get_tank_render_container(self.project.name,
                                                   self.scene.name,
                                                   self.name,
                                                   department)



        return tank.local.Em().find( "%(revision_type)s(%(version)s, %(render_container)s)" % vars() )

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



    def shotgun_field(self, fields):
        """
        Return the shotgun fields related to the shot.
        ex:
        shot.shotgun_field(['code','type','id','sg_handle_start'])
        {'sg_handle_start': 997, 'code': '21a_010', 'type': 'Shot', 'id': 3849}
        """
        # first check if data is already all cache
        field_name_list = [fields] if type(fields) == str else fields

        field_value_list = [ self._shotgun_fields[field_name] for field_name in field_name_list if self._shotgun_fields.has_key(field_name) ]

        if len(field_value_list) != len(field_name_list):
            result = shotgun_session().get_shot(self.scene.name, self.name, fields=field_name_list)

            # now cache the read values
            for k in result.keys():
                self._shotgun_fields[k] = result[k]

            return result[fields] if type(fields)==str else result
        else:
            return field_value_list[0] if len(field_value_list)==1 else dict(zip( field_name_list, field_value_list))

    def list_all_shotgun_fields(self):
        """
        Return all the possible fields of a Shotgun Shot
        """
        return shotgun_session().api().schema_field_read("Shot").keys()


    def get_cut_order(self, as_string=False):
        '''
        Return the order tuple, or zero padded as string
        '''
        if self.shotgun_field("sg_cut_order") == None:
            self._shotgun_fields["sg_cut_order"] = self.scene.generate_shot_order(self.name)

        if as_string:
            return ("%0"+str(ORDER_ID_ZERO_PADDING+1)+ "d_%0"+ str(ORDER_ID_ZERO_PADDING+1) +"d") % \
                    (self.scene.shotgun_field("sg_sort_order"), self.shotgun_field("sg_cut_order"))

        else:
            return (self.scene.shotgun_field("sg_sort_order"), self.shotgun_field("sg_cut_order"))


ORDER_ID_ZERO_PADDING = 6   # the number of zero to pad scene or shot that doesn't have a order
class Scene:
    @staticmethod
    def _construct(scene, project=None):
        if isinstance(scene, Scene):
            return scene
        else:
            project_obj = Project._construct(project)
            return Scene(scene=scene, project = project_obj )

    def __init__(self, scene, project=None, shotgun_fields={}):
        self.project    = Project._construct(project)
        self.name       = scene

        self._shotgun_fields    = shotgun_fields
        self._cache_shot_list   = None
        self._cache_shot_render = None


    def shotgun_field(self, fields):
        """
        Return the shotgun fields related to the scene.
        ex:
        shot.shotgun_field(['code','type','id','sg_handle_start'])
        {'sg_handle_start': 997, 'code': '21a_010', 'type': 'Shot', 'id': 3849}
        """

        # first check if data is already all cache
        field_name_list = [fields] if type(fields) == str else fields

        field_value_list = [ self._shotgun_fields[field_name] for field_name in field_name_list if self._shotgun_fields.has_key(field_name) ]

        if len(field_value_list) != len(field_name_list):
            result = shotgun_session().get_scene(self.name, fields=field_name_list)

            # now cache the read values
            for k in result.keys():
                self._shotgun_fields[k] = result[k]

            return result[fields] if type(fields)==str else result
        else:
            return field_value_list[0] if len(field_value_list)==1 else dict(zip( field_name_list, field_value_list))


    def list_all_shotgun_fields(self):
        """
        Return all the possible fields of a Shotgun Shot
        """
        return shotgun_session().api().schema_field_read("Scene").keys()


    def get_all_scene_render_shotgun(self, reload=False):
        if not reload and self._cache_shot_render:
            return self._cache_shot_render

        self._cache_shot_render =  self._get_render_data_shotgun()

        return self._cache_shot_render


    def _get_render_data_shotgun(self, search_spec=None, result_spec=None):
        if result_spec==None:
            result_spec =  ['department','sg_tank_address', 'sg_status_1', 'created_at', 'user','sg_rv_preview_link',
                            'sg_comments','sg_status_1']

        if search_spec==None:
            search_spec = [     ['sg_tank_address','contains','Scene(%s)'%self.name],
                                ['sg_tank_address','contains','ShotRender'],
                                ["sg_status_list","is_not", "omt"]]

        all_render = shotgun_session().api().find("Version",
                                    search_spec,
                                    result_spec,
                                    order = [{'field_name':'created_at','direction':'desc'}]
                                   )

        all_render_aux = [ {"aux": projectAwareness.determine_aux_data_from_tank_address_fast(r["sg_tank_address"]), "sg_data":r}
                          for r in all_render if r["sg_tank_address"]!=None]

        return self._organize_render_data(all_render_aux)


    def _organize_render_data(self, shotgun_raw_version_data):
        '''
        Data is organized by department, follow by shot
        '''
        # only take the creative ones
        shotgun_raw_version_data = [ r for r in shotgun_raw_version_data
                                        if r["aux"]["review_type"].lower() in (
                                                                                "technical",
                                                                                "creative",
                                                                                ) ]

        # organize the data
        all_scene_movie_hash = {}

        for render in shotgun_raw_version_data:
            dept = render["aux"]["department"]
            shot = render["aux"]["shot"]
            if not all_scene_movie_hash.has_key(dept):
                all_scene_movie_hash[dept] = {}

            dept_hash = all_scene_movie_hash[dept]

            if not dept_hash.has_key(shot):
                dept_hash[shot] = []

            dept_hash[shot].append(
                                   {
                                    'address':  render["sg_data"]["sg_tank_address"],
                                    'scene':    render["aux"]["scene"],
                                    'shot':     render["aux"]["shot"],
                                    'name':     render["aux"]["rev_name"],
                                    'date':     render["sg_data"]["created_at"],
                                    'sg_data':  render["sg_data"],
                                    'review_type':  render["aux"]["review_type"].lower(),
                                    }
                                   )

        # now sort the shots
        for dept in all_scene_movie_hash.keys():
            dept_hash = all_scene_movie_hash[dept]

            for shot in dept_hash.keys():
                mov_list = dept_hash[shot]
                mov_list.sort(_sort_shotgun_clips)

                # some movies are not publish properly
                # iterate through the all the revision from latest to oldest
                # remove any invalid movies until the first valid one is reached.
#                for mov in mov_list:
#                    tank_obj = get_tank_obj_from_rv_link(mov['sg_data']['sg_rv_preview_link'])
#
#                    if tank_obj and is_valid_render(tank_obj.get_filesystem_location(), "Movie", check_filesize=True):
#                        break # first valide movie found break
#                    else:
#                        mov_list.remove(mov)

        return all_scene_movie_hash


    def get_all_shot_render_fast(self, shot_name, department=None,cachedOnly=False):
        # check to see if we're only loading data from the cache
        if ( not cachedOnly ):
            all_renders = self.get_all_scene_render_shotgun()
        
        if ( department and self._cache_shot_render ):
            dcache = self._cache_shot_render.get(department)
            if ( dcache ):
                output = dcache.get(shot_name)
                if ( output ):
                    return output
        return []
        
    def get_all_scene_movie_fast(self, render_type = "Not Used"):
        print "Deprecated: please use get_all_scene_render_shotgun"
        return self.get_all_scene_render_shotgun()


    def get_shot(self, shot_name, no_exception=False):
        shots = self.list_shots()
        shot_name = projectAwareness.format_shotgun_shot_name(shot_name = shot_name,
                                                              scene_name = self.name,
                                                              project_name = self.project.name)

        shot_search = [ shot for shot in shots if shot.name == shot_name ]

        if not shot_search:
            msg = "Scene %s does not have shot %s, available shots are {%s}" % (
                                                                                          self.name,
                                                                                          shot_name,
                                                                                          ",".join( [shot.name for shot in shots])
                                                                                          )

            if no_exception:
                print "Warning:", msg
            else:
                raise KeyError, msg


            return None

        return shot_search[0]




    def list_shots(self, ShotClass=None):
        '''
        List the scene in shotgun scene order
        '''
        if not ShotClass: ShotClass = Shot

        if self._cache_shot_list==None:
            all_shot_data           = shotgun_session().list_shots(self.name, fields=[   'id',
                                                                                'code',
                                                                                'sg_cut_order',
                                                                                'sg_handle_start',
                                                                                'sg_handle_end',
                                                                                'sg_cut_start',
                                                                                'sg_cut_end',
                                                                                'sg_status_list'])
            self._cache_shot_list   = [   ShotClass(
                                              shot              = data["code"],
                                              scene             = self,
                                              project           = self.project,
                                              shotgun_fields    = data,
                                              ) for data in all_shot_data ]

            # if the order is none, set it to 10000 + the int( name )
            for shot in self._cache_shot_list:
                if shot.shotgun_field("sg_cut_order") == None:
                    shot._shotgun_fields["sg_cut_order"] = self.generate_shot_order(shot.name)

            self._cache_shot_list.sort( lambda x,y: cmp(x.shotgun_field("sg_cut_order"), y.shotgun_field("sg_cut_order"))  )


        return self._cache_shot_list

    def generate_shot_order(self, shot_name):
        if "_" in shot_name: shot_name = shot_name.split("_")[1]

        shot_name_int = int( re.search("([0-9]+)",  shot_name ).groups()[0] )

        return pow(10, ORDER_ID_ZERO_PADDING) + shot_name_int

    def get_neighbour_shot(self, shot):
        shot_name = shot.name if isinstance(shot, Shot) else shot
        scene_name = self.name

        # search for the scene with the matchin name in the scene cache
        shots = self.list_shots()
        shot_obj_search = [ shot for shot in shots if shot.name == shot_name ]

        shot = shot_obj_search[0] if shot_obj_search else None

        if not shot: raise KeyError, "Scene '%(scene_name)s', Shot '%(shot_name)s' is not found in shotgun." % vars()

        # now find the neighbours
        index = shots.index(shot)

        prev = shots[index-1] if index!=0 else None
        next = shots[index+1] if index!=len(shots)-1 else None

        return prev, next

    def neighbours(self):
        return self.project.get_neighbour_scenes(self)


    def __repr__(self):
        return "Scene object: %s" % self.name


def get_tank_obj_from_rv_link(rv_link_string):
    # get the tank obj from the movie link
    result = re.search("\.(?P<entity_id>[0-9]+)_(?P<revision_id>[0-9]+)\.mov", rv_link_string)

    if result: # ie movie is not viewable through shotgun, hence don't show in review tool either

        tank_obj = tank.local.Em().get_revision_by_id(int(result.groupdict()["entity_id"]),
                                                      int(result.groupdict()["revision_id"]))
        return tank_obj


global _global_project
_global_project = None

def get_project(project=None):
    global _global_project
    if not _global_project:
        _global_project = Project(project, singletonOnly=True)

    return _global_project

class Project:
    def __init__(self, project = None, singletonOnly=False):
        project = project if project else os.environ.get("DRD_JOB")

        self.name = project

        self._cache_scene_list = None
        if not singletonOnly:
            raise IOError, "Please use get_project to ensure usage as singleton"

    def get_scene(self, scene):
        # get the scene name from parameter, which can either be a string, or a scene object
        scene_name = scene.name if isinstance(scene, Scene) else scene

        # search for the scene with the matchin name in the scene cache
        scenes = self.list_scenes()
        scene_obj_search = [ scene for scene in scenes if scene.name == scene_name ]

        scene = scene_obj_search[0] if scene_obj_search else None

        return scene


    def list_scenes(self, SceneClass=None):
        '''
        List the scene in shotgun scene order
        '''
        if not SceneClass: SceneClass = Scene

        if self._cache_scene_list == None:
            all_scene_data = shotgun_session().list_scenes(fields=['id','code','sg_sort_order'])

            self._cache_scene_list = [  SceneClass(
                                               scene            = data["code"],
                                               project          = self,
                                               shotgun_fields   = data )
                                                    for data in all_scene_data ]

            # if the order is none, set it to 10000 + the int( name )
            for scene in self._cache_scene_list:
                if scene.shotgun_field("sg_sort_order") == None:
                    scene_name_int = int( re.search("([0-9]+)", scene.name).groups()[0] )
                    scene._shotgun_fields["sg_sort_order"] = 1000000 + scene_name_int

            self._cache_scene_list.sort( lambda x,y: cmp(x.shotgun_field("sg_sort_order"), y.shotgun_field("sg_sort_order"))  )

        return self._cache_scene_list

    def list_scene_names(self):
        return [s.name for s in self.list_scenes()]

    def get_neighbour_scenes(self, scene):
        # get the scene name from parameter, which can either be a string, or a scene object
        scene_name = scene.name if isinstance(scene, Scene) else scene

        # search for the scene with the matchin name in the scene cache
        scenes = self.list_scenes()
        scene_obj_search = [ scene for scene in scenes if scene.name == scene_name ]

        scene = scene_obj_search[0] if scene_obj_search else None
        if not scene: raise KeyError, "Scene with name '%s' is not found in shotgun." % scene_name

        # now find the neighbours
        index = scenes.index(scene)

        prev = scenes[index-1] if index!=0 else None
        next = scenes[index+1] if index!=len(scenes)-1 else None

        return prev, next


    def __repr__(self):
        return "Project object: %s" % self.name


    @staticmethod
    def _construct(project):
        """ create the project context object """
        return project if isinstance(project, Project) else Project(project)

def _sort_shotgun_clips(y, x):
    # force techincal reviews to the bottom
    if x['review_type'] == 'creative' and y['review_type']=='technical':
        return 1

    elif y['review_type']== 'creative' and x['review_type']=='technical':
        return -1

    # sort by the status if set
    elif ( SORT_METHOD == 'sort_by_status' ):
        statusX = SORT_STATUS_ORDER.get(x["sg_data"]["sg_status_1"],50000)  # force empty statuses to the bottom
        statusY = SORT_STATUS_ORDER.get(y["sg_data"]["sg_status_1"],50000)  # force empty statuses to the bottom
        if ( statusX != statusY ):
            return cmp(statusY,statusX)
           
    # by default, sort by date and name
    result = cmp(x["date"], y["date"])
    if result ==0:
        return cmp(x['name'], y['name'])
    else:
        return result


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

