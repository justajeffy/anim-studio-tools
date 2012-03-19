

def test_context():
    #print '.....900 id',Scene("19d").shotgun_field("id")

    #print '......best cut', Cut(scene='19d').get_best_cut()
    #print '.......id', Shot(shot='21a_010',scene="21a").get_cut_info()

#    shot = Shot(shot='21a_010',scene="21a")
#    print '.......id', shot.get_cut_info()
#    print '.......render', shot.get_render()
#
#    prj = Project()
#    shot = prj.get_scene("19d").get_shot("19d_010")
#    prev, next = shot.neighbours()
#    if prev: print 'prev render ', prev.get_render()
#    if next: print 'next render ', next.get_render()

    shot = Project().get_scene("21a").get_shot("21a_010")
    print '......shot', shot.shotgun_field("code")

    start = time()
    print '...shot cut info', shot, ' ', shot.get_cut_info()
    print time() - start
#    sc = Project().get_scene('900')
#    print sc, ' .....' , sc.shotgun_field(["code"])
#    print '....neighbour scene', sc.neighbours()
#
#    print '......shot', sc.get_shot('900_001')


#    pc = Project()
#
#    print '......all scenes', pc.list_scenes()
#    print '........neighbour scene to 900', pc.get_neighbour_scenes('900')

    #print '..........neighbours of 10b', sc.neighbours()
#    sc = Scene('21a')
#    print sc.list_shots()
#
#    print '......shot 21a 010', sc.get_shot('21a_010')

    #shot = Shot('21a','21a_020')
    #print '.......', shot.neighbours()
    #print shotgun_session().api().schema_field_read('Scene').keys()

    #print Scene('21a').shotgun_field( ['code', 'id','type'])

    #print '.......read from cache'

    #print Scene('21a').shotgun_field( ['code'])




    #print 'neightbour', Shot('21b','21a_010').neighbours()





#    pscene = [ data for data in pc.list_scenes() if data.name=="900"][0]
#
#
#    #print '........pscene ', pscene
#    print pscene.list_shots()
    #print Scene('900').list_shots()


TEST_LIMIT = 10

def compare_vfs_and_file_system_id():
    count= 0
    for item in tank.server.Em().find("Movie").get_children():
        if count> TEST_LIMIT:
            break
        count+=1
        tank_obj = tank.find(str(item))

#        vfs_path = tank_obj.system.vfs_full_paths[0]
#        filesys_location = tank_obj.system.filesystem_location

        try:
            vfs_path = tank_obj.system.vfs_full_paths[0]
            filesys_location = tank_obj.system.filesystem_location
        except:
            print "warning can not resolved paths for %s" % str(item)
            continue

        if not vfs_path.endswith("mov"): continue


        cid, rid = re.search("\.([0-9]+)_([0-9]+)\.", vfs_path).groups()
        cid2, rid2 = re.search("/([0-9]+)/r([0-9]+)_", filesys_location).groups()

        if cid!=cid2 or rid!=rid2:
            print 'vfs path', vfs_path

            print cid, rid
            print cid2, rid2
        else:
            print '.',


def test():

    # get the neighbour movies
    from ContextSession import get_project
    from pprint import pprint

    p = get_project()
    d = p.get_scene("21a").get_all_shot_render_fast("21a_030","anim")

    pprint(d)




if __name__== "__main__":
    test()

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

