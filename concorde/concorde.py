#!/usr/bin/env python2.5


#-----------------------------------------------------------------------------
import os, re, string, sys, types, subprocess, math, pimath, ui, time, vacpy
from grind.util.glWidget import GLWidget
from grind.util.camera import Camera

from rodin import logging
from PyQt4 import QtGui, QtCore, uic
log = logging.get_logger('grind.concorde')

import grind
import BEE

try:
    from OpenGL.GLUT import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
except:
    print ''' Error: PyOpenGL not available !!'''
    sys.exit()

class VacCamInfos:
    def __init__(self,frame):
        self.frame = frame
        self.focal_distance = 0
        self.focal_length = 0
        self.horizontal_aperture = 0
        self.vertical_aperture = 0
        self.near_clip = 0
        self.far_clip = 0
        self.fov = 0
        self.fovy = 0
        self.matrix = None

#-----------------------------------------------------------------------------

def get_path( p ):
    return os.path.join(os.path.dirname(__file__),p)

def getHumanReadableMemAsStr(num):
    for x in ['bytes','KB','MB','GB','TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def comma_me(amount):
    orig = amount
    new = re.sub("^(-?\d+)(\d{3})", '\g<1>,\g<2>', amount)
    if orig == new:
        return new
    else:
        return comma_me(new)

def formatMemoryStr():
    freeMem = grind.gpu_free_mem()
    totalMem = grind.gpu_total_mem()
    if totalMem == 0:
        print "No memory available !\nSomething's wrong with your GPU !\nClosing Concorde now.."
        sys.exit(0)
    percent = ( float(freeMem) / float(totalMem) ) * 100.0
    d = 1024*1024
    return "GPU Mem Free: %2.1f%% (%s/%s)" % (percent,getHumanReadableMemAsStr(freeMem),getHumanReadableMemAsStr(totalMem))

def int_to_str(i):
    # dirty I know...
    s=''
    m= 1000
    while i < m:
        s+='0'
        m = m / 10
    return s+str(i)

ps_stack = []
ps_stack_maxsize = 0

#-----------------------------------------------------------------------------
class PtcObject:
#-----------------------------------------------------------------------------
    def __init__(self, filename, density, autoComputeDensity=True):
        self.fullfilename = filename
        self.filename = os.path.basename( self.fullfilename )
        self.ps = grind.ParticleSystem()

        autoComputeDensity = False
        if autoComputeDensity and density == 100:
            ndensity = density
            filesize = os.path.getsize(filename)
            if filesize > 500 * 1024 * 1024: ndensity = 10
            elif filesize > 100 * 1024 * 1024: ndensity = 50
            if ndensity != density: print "Density forced to %d %% (file too big)" % ndensity
            density = ndensity

        freeMem = grind.gpu_free_mem()
        self.gpuSize = self.ps.get_gpu_size_of(self.fullfilename, density)
        while self.gpuSize >= freeMem:
            density = density / 2
            self.gpuSize = self.ps.get_gpu_size_of(self.fullfilename, density)

        startTime = time.time()
        self.ps.read(self.fullfilename, density)
        endTime = time.time()
        print '\nLoaded in ' + str('%.2f'%(endTime - startTime)) + 's'

        ivpMtx = self.ps.get_inv_view_proj_matrix()
        self.n0 = ivpMtx.multVecMatrix( pimath.V3f( 1, 1, 0) )
        self.n1 = ivpMtx.multVecMatrix( pimath.V3f(-1, 1, 0) )
        self.n2 = ivpMtx.multVecMatrix( pimath.V3f( 1,-1, 0) )
        self.n3 = ivpMtx.multVecMatrix( pimath.V3f(-1,-1, 0) )
        self.f0 = ivpMtx.multVecMatrix( pimath.V3f( 1, 1, 1) )
        self.f1 = ivpMtx.multVecMatrix( pimath.V3f(-1, 1, 1) )
        self.f2 = ivpMtx.multVecMatrix( pimath.V3f( 1,-1, 1) )
        self.f3 = ivpMtx.multVecMatrix( pimath.V3f(-1,-1, 1) )

        self.density = density

        self.enabled = True
        if ps_stack_maxsize > 0:
            if len(ps_stack) >= ps_stack_maxsize: del ps_stack.pop()[1]
            ps_stack.insert(0, [self.fullfilename, self.ps] )

        s = self.filename.split('.')
        ext = s[len(s)-1]
        frame = s[len(s)-2]
        if frame.isdigit():
            self.loaded_frame = int(frame)
        else:
            self.loaded_frame = -1
        self.name = s[0]

        self.min_range = self.loaded_frame
        self.max_range = self.loaded_frame
        self.available_frames = []
        if self.loaded_frame == -1: return

        all_files = os.popen( "ls " + self.fullfilename.replace(str(self.loaded_frame), "*") ).read().split('\n')

        for f in all_files:
            if f != "":
                s = f.split('.')
                ext = s[len(s)-1]
                frame = s[len(s)-2]
                if frame.isdigit():
                    iframe = int(frame)
                    self.available_frames.append( iframe )
                    self.min_range = min(self.min_range, iframe)
                    self.max_range = max(self.max_range, iframe)

    def can_use_slider(self):
        return len(self.available_frames) == ( self.max_range - self.min_range + 1 )

    def contains(self, fullfilename):
        filename = os.path.basename( fullfilename )
        return filename.split('.')[0] == self.name

    def load_specific_frame(self, current_frame):
        newfullfilename = self.fullfilename.replace(int_to_str(self.loaded_frame), int_to_str(current_frame))
        if newfullfilename == self.fullfilename: return
        if os.path.isfile(newfullfilename) == True:
            self.fullfilename = newfullfilename
            self.loaded_frame = current_frame
            for s in ps_stack:
                if s[0] == self.fullfilename:
                    self.ps = s[1]
                    return
            del self.ps
            self.ps = grind.ParticleSystem()
            self.ps.read(self.fullfilename, self.density)
            if ps_stack_maxsize > 0:
                if len(ps_stack) >= ps_stack_maxsize: del ps_stack.pop()[1]
                ps_stack.insert(0, [self.fullfilename, self.ps] )

    def reload_frame(self, current_frame):
        if current_frame == self.loaded_frame: return
        load_next = True if ( current_frame > self.loaded_frame ) else False
        idx = self.available_frames.index(self.loaded_frame)
        if load_next and (idx+1 < len(self.available_frames)):
            current_frame = self.available_frames[idx+1]
        elif not load_next and (idx-1) >= 0:
            current_frame = self.available_frames[idx-1]
        newfullfilename = self.fullfilename.replace(int_to_str(self.loaded_frame), int_to_str(current_frame))

        self.fullfilename = newfullfilename
        self.loaded_frame = current_frame
        for s in ps_stack:
            if s[0] == self.fullfilename:
                self.ps = s[1]
                return
        del self.ps
        self.ps = grind.ParticleSystem()
        self.ps.read(self.fullfilename, self.density)
        if ps_stack_maxsize > 0:
            if len(ps_stack) >= ps_stack_maxsize: del ps_stack.pop()[1]
            ps_stack.insert(0, [self.fullfilename, self.ps] )

#-----------------------------------------------------------------------------
class PtcRenderer:
#-----------------------------------------------------------------------------
    def __init__(self):
        self.ptcobj_array = []

        self.framerateLabel = None
        self.previous_curtime = 0

        self.renderer = BEE.Renderer()
        initLutStr = self.renderer.initLut()
        lutGlslFunc = "#version 120 \n" + initLutStr

        BEE.ProgramUseIncludeString( lutGlslFunc )

        self.point_shader = BEE.Program()
        self.point_shader.read( os.path.join(get_path(''), 'glsl/point.vs.glsl'),
                                os.path.join(get_path(''), 'glsl/point.fs.glsl'))

        BEE.ProgramUseIncludeString( None )

        lutGlslFunc = "#version 120 \n" + "#extension GL_EXT_geometry_shader4 : enable \n" + initLutStr
        BEE.ProgramUseIncludeString( lutGlslFunc )

        self.disc_shader = BEE.Program()
        self.disc_shader.read( os.path.join(get_path(''), 'glsl/disc.vs.glsl'),
                               os.path.join(get_path(''), 'glsl/disc.fs.glsl'),
                               os.path.join(get_path(''), 'glsl/disc.gs.glsl'),
                               GL_POINTS,
                               GL_TRIANGLE_STRIP)

        self.display_vector = False
        self.display_vector_shader = BEE.Program()
        self.display_vector_shader.read( os.path.join(get_path(''), 'glsl/display_vector.vs.glsl'),
                                         os.path.join(get_path(''), 'glsl/display_vector.fs.glsl'),
                                         os.path.join(get_path(''), 'glsl/display_vector.gs.glsl'),
                                         GL_POINTS,
                                         GL_LINE_STRIP)

        BEE.ProgramUseIncludeString( None )

        self.default_shader = BEE.Program()
        self.default_shader.read( os.path.join(get_path(''), 'glsl/default.vs.glsl'),
                                  os.path.join(get_path(''), 'glsl/default.fs.glsl'))

        self.scene_bounds = grind.BBox()

        self.current_shader = self.point_shader
        self.view_back_faces = False
        self.bbox_display = False
        self.shot_frustum_display = False
        self.bake_frustum_display = False
        self.lighting = True # init inverted for some reason...
        self.disc_point_distance_transition = 50
        self.keepCameraPositionOnAddingPtc = False

        self.useWipe = False
        self.wipe_values = pimath.V4f(-1, 1, -1, 1) # default view everything !
        self.hSliderValue = 0.0
        self.vSliderValue = 0.0

        self.user_data_channel_index = "None"
        self.user_data_channel_index_for_vector_display = "None"
        self.display_position = False
        self.display_normal = False
        self.display_radius = False
        self.radius_factor = 1
        self.exposure = 0

        # setup some gl states
        glCullFace( GL_BACK );

    def update_bounds(self, upbds):
        if upbds.min.x < self.scene_bounds.min.x:
             self.scene_bounds.min.x = upbds.min.x
        if upbds.min.y < self.scene_bounds.min.y:
             self.scene_bounds.min.y = upbds.min.y
        if upbds.min.z < self.scene_bounds.min.z:
             self.scene_bounds.min.z = upbds.min.z
        if upbds.max.x > self.scene_bounds.max.x:
             self.scene_bounds.max.x = upbds.max.x
        if upbds.max.y > self.scene_bounds.max.y:
             self.scene_bounds.max.y = upbds.max.y
        if upbds.max.z > self.scene_bounds.max.z:
             self.scene_bounds.max.z = upbds.max.z

    def clearAll(self):
        self.ptcobj_array = []

    def add_ptc(self, filename, density, id):
        ptcObj = PtcObject(filename, density)
        self.bake_cam_position = ptcObj.ps.get_bake_cam_position()
        self.bake_cam_look_at = ptcObj.ps.get_bake_cam_look_at()
        self.bake_cam_up = ptcObj.ps.get_bake_cam_up()
        self.bake_view_proj_matrix = ptcObj.ps.get_view_proj_matrix()
        ptcObj.id = id
        self.ptcobj_array.append( ptcObj )
        self.update_bounds( ptcObj.ps.get_bounds() )
        return ptcObj

    def getBounds(self):
        return self.scene_bounds

    def use_point_shader(self):
        self.current_shader = self.point_shader

    def use_disc_shader(self):
        self.current_shader = self.disc_shader

    def toggle_view_back_faces(self):
        self.view_back_faces = not self.view_back_faces

    def toggle_bbox_display(self):
        self.bbox_display = not self.bbox_display

    def toggle_bake_frustum_display(self):
        self.bake_frustum_display = not self.bake_frustum_display

    def toggle_shot_frustum_display(self):
        self.shot_frustum_display = not self.shot_frustum_display

    def toggle_lighting(self):
        self.lighting = not self.lighting

    def toggle_useWipe(self):
        self.useWipe = not self.useWipe
        return self.useWipe

    def toggleKeepCameraPositionOnAddingPtc(self):
        self.keepCameraPositionOnAddingPtc = not self.keepCameraPositionOnAddingPtc

    def getAllPtcInfos(self):
        info_list = []
        for ptcObj in self.ptcobj_array:
            info_list.append( [ptcObj.fullfilename, ptcObj.density] )
        return info_list

    def getAllInvisiblePtcInfos(self):
        info_list = []
        for ptcObj in self.ptcobj_array:
            if ptcObj.enabled == False:
                info_list.append( [ptcObj.fullfilename, ptcObj.density] )
        return info_list

    def getVisiblePtcInfosCount(self):
        count = 0
        for ptcObj in self.ptcobj_array:
            if ptcObj.enabled == True: count = count + 1
        return count

    def getUserDataNameList(self):
        name_list = []
        for ptcObj in self.ptcobj_array:
            ps = ptcObj.ps
            for i in range( ps.get_user_data_count() ):
                if name_list.count( ps.get_user_data_name(i) ) == 0:
                    name_list.append( ps.get_user_data_name(i) )
        return name_list

    def getVectorTypeUserDataNameList(self):
        name_list = []
        for ptcObj in self.ptcobj_array:
            ps = ptcObj.ps
            for i in range( ps.get_user_data_count() ):
                if ps.get_user_data_type(i) == "vector" or ps.get_user_data_type(i) == "normal":
                    if name_list.count( ps.get_user_data_name(i) ) == 0:
                        name_list.append( ps.get_user_data_name(i) )
        return name_list

    def use_user_data_channel_index(self, chname):
        self.display_position = False
        self.display_normal = False
        self.display_radius = False

        self.user_data_channel_index = chname

        if chname == "None":
            return

        if chname == "Position":
            self.display_position = True
            return

        if chname == "Normal":
            self.display_normal = True
            return

        if chname == "Radius":
            self.display_radius = True
            return

    def use_user_data_channel_index_for_vector_display(self, chname):
        self.display_normal_for_vector_display = False
        self.display_vector = True

        self.user_data_channel_index_for_vector_display = chname

        if chname == "None":
            self.display_vector = False
            return

        if chname == "Normal":
            self.display_normal_for_vector_display = True
            return

    def set_radius_factor(self, radius_factor):
        self.radius_factor = radius_factor

    def set_current_frame(self, current_frame, auto=False):
        ret = -1
        for ptcObj in self.ptcobj_array:
            if ptcObj.min_range != ptcObj.max_range and ptcObj.min_range <= current_frame and ptcObj.max_range >= current_frame:
                if auto: ptcObj.reload_frame( current_frame )
                else: ptcObj.load_specific_frame( current_frame )
                ret = ptcObj.loaded_frame
        return ret

    def enable_ptc(self, name, id, enabled ):
        for ptcObj in self.ptcobj_array:
            if ptcObj.name == name and ptcObj.id == id:
                ptcObj.enabled = enabled

    def reload_with_density(self,density):
        new_ptcobj_array = []
        for ptcObj in self.ptcobj_array:
            if ptcObj.enabled == True:
                # we only reload the visible ptc
                if ptcObj.density != density:
                    fullfilename = ptcObj.fullfilename
                    print "Reloading file.. " + fullfilename
                    id = ptcObj.id
                    newPtcObj = PtcObject(fullfilename, density, False)
                    newPtcObj.id = id
                    new_ptcobj_array.append(newPtcObj)
        self.ptcobj_array = new_ptcobj_array

    def update(self):
        pass

    def render_ps_with_shader(self,ps,ambient,shader,disc_point_distance_transition):
        shader.use()

        BEE.useLutTextureOnProgram(self.renderer, shader, 0)

        shader.setUniform("ambient", ambient, ambient, ambient)
        shader.setUniform("display_position", self.display_position)
        shader.setUniform("display_normal", self.display_normal)
        shader.setUniform("display_radius", self.display_radius)
        shader.setUniform("channel_size", ps.get_current_user_data_size())
        shader.setUniform("radius_factor", self.radius_factor * 1.45)
        shader.setUniform("exposure", self.exposure)
        shader.setUniform("disc_point_distance_transition", disc_point_distance_transition)
        shader.setUniform("wipe_values", self.wipe_values.x, self.wipe_values.y, self.wipe_values.z, self.wipe_values.w )
        # we have to do the backface culling in the shader
        # as the hardware can't backface a point ! ;)
        shader.setUniform("backface_cull", self.view_back_faces)

        ps.render(1)
        shader.release()

    def render_ps(self, ptcObj, ambient):
        ps = ptcObj.ps
        if ps.get_real_point_count() == 0:
            return

        if self.bake_frustum_display:
            self.default_shader.use()
            self.default_shader.setUniform("ambient", 0, 1, 0)
            self.default_shader.setUniform("wipe_values", self.wipe_values.x, self.wipe_values.y, self.wipe_values.z, self.wipe_values.w )
            self.display_frustum(ptcObj.n0, ptcObj.n1, ptcObj.n2, ptcObj.n3, ptcObj.f0, ptcObj.f1, ptcObj.f2, ptcObj.f3)
            self.default_shader.release()

        ps.set_current_user_data_index( -1 )
        for i in range( ps.get_user_data_count() ):
            if self.user_data_channel_index == ps.get_user_data_name(i):
                ps.set_current_user_data_index( i )

        if self.current_shader == self.disc_shader:
            self.render_ps_with_shader(ps, ambient, self.point_shader, self.disc_point_distance_transition)
            self.render_ps_with_shader(ps, ambient, self.disc_shader, self.disc_point_distance_transition)
        else:
            self.render_ps_with_shader(ps, ambient, self.point_shader, 0)

        if self.display_vector == True:
            ps.set_current_user_data_index( -1 )
            for i in range( ps.get_user_data_count() ):
                if self.user_data_channel_index_for_vector_display == ps.get_user_data_name(i):
                    ps.set_current_user_data_index( i )

            self.display_vector_shader.use()
            self.display_vector_shader.setUniform("ambient", ambient, ambient, ambient)
            self.display_vector_shader.setUniform("display_normal", self.display_normal_for_vector_display)
            self.display_vector_shader.setUniform("radius_factor", self.radius_factor)
            self.display_vector_shader.setUniform("exposure", self.exposure)
            self.display_vector_shader.setUniform("backface_cull", self.view_back_faces) # we have to do it in the shader
            self.display_vector_shader.setUniform("wipe_values", self.wipe_values.x, self.wipe_values.y, self.wipe_values.z, self.wipe_values.w )
            ps.render(1)
            self.display_vector_shader.release()

    def display_frustum(self,n0, n1, n2, n3, f0, f1, f2, f3):
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glBegin(GL_QUADS)
        glVertex3d(n0.x, n0.y, n0.z)
        glVertex3d(n1.x, n1.y, n1.z)
        glVertex3d(n3.x, n3.y, n3.z)
        glVertex3d(n2.x, n2.y, n2.z)
        glVertex3d(f0.x, f0.y, f0.z)
        glVertex3d(f1.x, f1.y, f1.z)
        glVertex3d(f3.x, f3.y, f3.z)
        glVertex3d(f2.x, f2.y, f2.z)
        glEnd()

        glBegin(GL_LINES)
        glVertex3d(n0.x, n0.y, n0.z)
        glVertex3d(f0.x, f0.y, f0.z)
        glVertex3d(n1.x, n1.y, n1.z)
        glVertex3d(f1.x, f1.y, f1.z)
        glVertex3d(n3.x, n3.y, n3.z)
        glVertex3d(f3.x, f3.y, f3.z)
        glVertex3d(n2.x, n2.y, n2.z)
        glVertex3d(f2.x, f2.y, f2.z)
        glEnd()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def render(self):
        curtime = time.clock()
        diff = ( curtime - self.previous_curtime )
        if (diff > 0):
            diff = 1.0 / diff
            self.framerateLabel.setText('FPS = ' + str(int(diff)))
        else:
            self.framerateLabel.setText('FPS = MAX')
        self.previous_curtime = curtime

        ambient = 0.05
        if self.lighting == True: ambient = 1

        if self.shot_frustum_display:
            self.default_shader.use()
            self.default_shader.setUniform("ambient", 1, 0, 0)
            self.default_shader.setUniform("wipe_values", -1, 1, -1, 1 ) # turn off wipe
            self.display_frustum(self.n0, self.n1, self.n2, self.n3, self.f0, self.f1, self.f2, self.f3)
            self.default_shader.release()

        count = 0
        for ptcObj in self.ptcobj_array:
            if ptcObj.enabled == True:
                self.wipe_values = pimath.V4f(-1, 1, -1, 1) # default view everything !
                if self.useWipe:
                    if count == 0:
                        if self.hSliderValue != 0.0: self.wipe_values.y = self.hSliderValue
                        elif self.vSliderValue != 0.0: self.wipe_values.w = self.vSliderValue
                    else:
                        if self.hSliderValue != 0.0: self.wipe_values.x = self.hSliderValue
                        elif self.vSliderValue != 0.0: self.wipe_values.z = self.vSliderValue
                self.render_ps(ptcObj, ambient)
                count = count + 1

        glDisable(GL_CULL_FACE)

        # render bbox in wireframe
        if self.bbox_display == True:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            self.default_shader.use()
            self.default_shader.setUniform("ambient", 0, 0, 1)
            self.getBounds().render(1)
            self.default_shader.release()
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

#-----------------------------------------------------------------------------
class ConcordeWindow(ui.ConcordeWindow):
    def __init__(self,fileList_to_load=None):
        ui.ConcordeWindow.__init__(self)

        self.connect(self.actionFileNew, QtCore.SIGNAL('triggered()'), self.onActionFileNewTriggered)
        self.connect(self.actionFileAdd, QtCore.SIGNAL('triggered()'), self.onActionFileAddTriggered)
        self.connect(self.actionFileClose, QtCore.SIGNAL('triggered()'), self.onActionFileCloseTriggered)
        self.connect(self.actionFileRefresh, QtCore.SIGNAL('triggered()'), self.onActionFileRefreshTriggered)
        self.connect(self.actionFileQuit, QtCore.SIGNAL('triggered()'), self.onActionFileQuitTriggered)
        self.connect(self.actionCameraReset, QtCore.SIGNAL('triggered()'), self.onActionCameraResetTriggered)
        self.connect(self.actionCameraShowShotView, QtCore.SIGNAL('triggered()'), self.onActionCameraShowShotViewTriggered)
        self.connect(self.actionCameraShowBakeView, QtCore.SIGNAL('triggered()'), self.onActionCameraShowBakeViewTriggered)
        self.connect(self.actionCameraUpdateFarClip, QtCore.SIGNAL('triggered()'), self.onActionCameraUpdateFarClipTriggered)
        self.connect(self.actionCameraTargetSelection, QtCore.SIGNAL('triggered()'), self.onActionCameraTargetSelectionTriggered)
        self.connect(self.actionCameraResetExposure, QtCore.SIGNAL('triggered()'), self.onActionCameraResetExposureTriggered)
        self.connect(self.actionCameraResetDiscPointTransitionDistance, QtCore.SIGNAL('triggered()'), self.onActionCameraResetDiscPointTransitionDistanceTriggered)
        self.connect(self.actionCameraLoadVacFrameInfos, QtCore.SIGNAL('triggered()'), self.onActionCameraLoadVacFrameInfosTriggered)


        self.connect(self.actionKeepCameraPositionOnAddingPtc, QtCore.SIGNAL('triggered()'), self.onActionKeepCameraPositionOnAddingPtcTriggered)
        self.connect(self.actionDisplayBBox, QtCore.SIGNAL('triggered()'), self.onActionDisplayBBoxTriggered)
        self.connect(self.actionDisplayBakeFrustum, QtCore.SIGNAL('triggered()'), self.onActionDisplayBakeFrustumTriggered)
        self.connect(self.actionDisplayShotFrustum, QtCore.SIGNAL('triggered()'), self.onActionDisplayShotFrustumTriggered)
        self.connect(self.actionViewLighting, QtCore.SIGNAL('triggered()'), self.onActionViewLightingTriggered)
        self.connect(self.actionViewBackFaces, QtCore.SIGNAL('triggered()'), self.onActionViewBackFacesTriggered)
        self.connect(self.actionUseWipe, QtCore.SIGNAL('triggered()'), self.onActionUseWipeTriggered)

        #self.connect(self.actionViewIncrPtcTreeDepth, QtCore.SIGNAL('triggered()'), self.onActionViewIncrPtcTreeDepth)
        #self.connect(self.actionViewDecrPtcTreeDepth, QtCore.SIGNAL('triggered()'), self.onActionViewDecrPtcTreeDepth)

        self.glWidget = GLWidget(self.left)
        self.hSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.vSlider = QtGui.QSlider(QtCore.Qt.Vertical)

        gl = QtGui.QGridLayout()
        gl.addWidget(self.vSlider, 0, 0)
        gl.addWidget(self.glWidget, 0, 1)
        gl.addWidget(self.hSlider, 1, 1)
        gl.setSpacing(1)
        gl.setMargin(1)
        self.left.setLayout(gl)
        self.hSlider.hide(); self.vSlider.hide()
        self.hSlider.setValue(49); self.vSlider.setValue(49) # 49 + 1 = 50 :)

        self.connect(self.hSlider, QtCore.SIGNAL('valueChanged(int)'), self.onValueChangedHSlider)
        self.connect(self.vSlider, QtCore.SIGNAL('valueChanged(int)'), self.onValueChangedVSlider)

        self.loadSettings()
        self.setAcceptDrops( True )

        grind.info() # lets now init grind/bee/ogl

        # force the first frame to be cleared !
        glClearColor (0, 0, 0, 0)
        self.glWidget.updateGL()

        self.ptcRenderer = None
        self.glWidget.drawGrid = False
        self.glWidget.drawDefaultObject = False
        self.glWidget.display_selection_marker = True # always display it

        self.statusMsg( "Please open a ptc file..." )

        self.connect(self.menuViewDensity, QtCore.SIGNAL('triggered(QAction*)'), self.onMenuViewDensity)

        # add ViewDisplayMode actions
        self.actionViewDisplayModePoints = QtGui.QAction("Points",self)
        self.actionViewDisplayModePoints.setCheckable(True)
        self.actionViewDisplayModePoints.setShortcut("P")
        self.menuViewDisplayMode.addAction(self.actionViewDisplayModePoints)
        self.toolBar.addAction(self.actionViewDisplayModePoints)

        self.actionViewDisplayModeDiscs = QtGui.QAction("Discs",self)
        self.actionViewDisplayModeDiscs.setCheckable(True)
        self.actionViewDisplayModeDiscs.setShortcut("D")
        self.menuViewDisplayMode.addAction(self.actionViewDisplayModeDiscs)
        self.toolBar.addAction(self.actionViewDisplayModeDiscs)

        self.actionViewDisplayModePoints.setChecked(True)
        self.connect(self.actionViewDisplayModePoints, QtCore.SIGNAL('triggered()'), self.onActionViewDisplayModePointsTriggered)
        self.connect(self.actionViewDisplayModeDiscs, QtCore.SIGNAL('triggered()'), self.onActionViewDisplayModeDiscsTriggered)

        self.previousTriggeredViewPointColorAction = None
        self.previousTriggeredViewDisplayVectorsAction = None
        self.previousTriggeredViewRadiusFactorAction = self.actionViewRadiusFactorX1

        self.connect(self.menuViewPointColor, QtCore.SIGNAL('triggered(QAction*)'), self.onMenuViewPointColorTriggered)
        self.connect(self.menuViewDisplayVectors, QtCore.SIGNAL('triggered(QAction*)'), self.onMenuViewDisplayVectorsTriggered)
        self.connect(self.menuViewRadiusFactor, QtCore.SIGNAL('triggered(QAction*)'), self.onMenuViewRadiusFactorTriggered)
        self.connect(self.menuViewPointClouds, QtCore.SIGNAL('triggered(QAction*)'), self.onMenuViewPointCloudsTriggered)

        self.exposureWidget = AttribFloatWidget(self.right,"Exposure", 0, {"min":-6.0,"max":6.0,"min_hard":-6.0,"max_hard":6.0}, self.onExposureSliderValueChanged)
        self.rightLayout.addWidget(self.exposureWidget)

        self.discPointTransitionDistanceWidget = AttribFloatWidget(self.right,"Disc/Point Distance", 50, {"min":0.0,"max":500.0,"min_hard":0.0,"max_hard":10000000.0}, self.onDiscPointTransitionDistanceSliderValueChanged)
        self.rightLayout.addWidget(self.discPointTransitionDistanceWidget)

        self.discRadiusFactorWidget = AttribFloatWidget(self.right,"Disc Radius Factor", 1, {"min":0.0,"max":10.0,"min_hard":0.0,"max_hard":10000000.0}, self.onDiscRadiusFactorSliderValueChanged)
        self.rightLayout.addWidget(self.discRadiusFactorWidget)

        self.framerateLabel = QtGui.QLabel("framerate")
        self.statusBar.addWidget(self.framerateLabel)

        self.memoryLabel = QtGui.QLabel("memory")
        self.statusBar.addWidget(self.memoryLabel)
        self.memoryLabel.setText( formatMemoryStr() )

        self.frameControlWidget = None
        self.cameraVacFrameControlAdded = False
        self.cameraVacFrameControlWidget = None
        self.vacCamInfosArray = []

        if fileList_to_load != None:
            i = 0
            density = 100
            for f in fileList_to_load:
                if '-c' in f:
                    global ps_stack_maxsize
                    ps_stack_maxsize =  ( int( f.replace("-c","") ) )
                elif '-d' in f:
                    density =  ( int( f.replace("-d","") ) )
                elif '-v' in f:
                    cam_vac_file =  f.replace("-v","")
                    if os.path.isfile(cam_vac_file) == True:
                        self.loadCameraVacFile(cam_vac_file)
                else:
                    self.openFile( f, density, True )

    def dragEnterEvent(self,event):
        # get rid off qstring null bytes !
        filename=''
        for c in str(event.mimeData().text()):
            if c in string.printable:
                filename = filename + c
        filename = filename.replace("file://","")

        ext = str(os.path.splitext( filename )[ 1 ]).lower()
        density = 100
        if ext.startswith( '.ptc' ):
            self.openFile( filename, density, True )
        else:
            print "Can't open " + filename + ' (wrong extension)'

    def onDiscPointTransitionDistanceSliderValueChanged(self,value):
        if self.ptcRenderer != None:
            self.ptcRenderer.disc_point_distance_transition = value
            self.glWidget.updateGL()

    def onExposureSliderValueChanged(self,value):
        if self.ptcRenderer != None:
            self.ptcRenderer.exposure = value
            self.glWidget.updateGL()

    def onDiscRadiusFactorSliderValueChanged(self,value):
        if self.ptcRenderer != None:
            self.ptcRenderer.set_radius_factor( value )
            self.glWidget.updateGL()

    def onFrameControlValueChanged(self,value,auto):
        if self.ptcRenderer != None:
            f = self.ptcRenderer.set_current_frame( value,auto )
            if f == -1 or (auto == False and f != value):
                self.statusMsg( "Missing Frame: " + str(value) )
                self.frameControlWidget.set_value( self.frameControlValue )
                return
            else:
                if f == self.frameControlValue: return
                self.statusMsg( "Loading Frame: " + str(f) )
            if f!=self.frameControlValue: self.frameControlWidget.set_value( f )
            self.memoryLabel.setText( formatMemoryStr() )
            self.glWidget.updateGL()
            self.frameControlValue = f

    def onMenuViewDensity(self,action):
        if self.ptcRenderer != None:
            density = ( int( action.text().remove("%") ) )
            self.ptcRenderer.reload_with_density( density )
            self.memoryLabel.setText( formatMemoryStr() )
            self.glWidget.updateGL()

    def onMenuViewPointCloudsTriggered(self,action):
        if self.ptcRenderer != None:
            t = action.text().split('.')
            if len(t) == 2: self.ptcRenderer.enable_ptc( t[0], int(t[1]), action.isChecked() )
            else: self.ptcRenderer.enable_ptc( action.text(), 0, action.isChecked() )
            count = self.ptcRenderer.getVisiblePtcInfosCount()
            if count == 2:
                self.hSlider.show(); self.vSlider.show()
                self.ptcRenderer.useWipe = self.actionUseWipe.isChecked()
                self.actionUseWipe.setEnabled(True)
            else:
                self.hSlider.hide(); self.vSlider.hide()
            	self.ptcRenderer.useWipe = False
                self.actionUseWipe.setEnabled(False)
            self.glWidget.updateGL()

    def onMenuViewPointColorTriggered(self,action):
        if self.previousTriggeredViewPointColorAction != None and self.previousTriggeredViewPointColorAction != action:
            self.previousTriggeredViewPointColorAction.setChecked(False)
            self.previousTriggeredViewPointColorAction = action
        if self.ptcRenderer != None:
            self.ptcRenderer.use_user_data_channel_index(action.text())
            self.glWidget.updateGL()

    def onMenuViewDisplayVectorsTriggered(self,action):
        if self.previousTriggeredViewDisplayVectorsAction != None and self.previousTriggeredViewDisplayVectorsAction != action:
            self.previousTriggeredViewDisplayVectorsAction.setChecked(False)
            self.previousTriggeredViewDisplayVectorsAction = action
        if self.ptcRenderer != None:
            self.ptcRenderer.use_user_data_channel_index_for_vector_display(action.text())
            self.glWidget.updateGL()

    def onMenuViewRadiusFactorTriggered(self,action):
        s = action.text().replace('x ','')
        if self.previousTriggeredViewRadiusFactorAction != None and self.previousTriggeredViewRadiusFactorAction != action:
            self.previousTriggeredViewRadiusFactorAction.setChecked(False)
            self.previousTriggeredViewRadiusFactorAction = action
        if self.ptcRenderer != None:
            self.ptcRenderer.set_radius_factor(float(s))
            self.glWidget.updateGL()

    def addViewPointColor(self,name):
        tmpAction = QtGui.QAction(name,self)
        tmpAction.setCheckable(True)
        self.menuViewPointColor.addAction(tmpAction)
        self.toolBarLeft.addAction(tmpAction)
        if self.previousTriggeredViewPointColorAction == None:
            self.previousTriggeredViewPointColorAction = tmpAction
        return tmpAction

    def addViewPointCloud(self,ptcobj):
        name = ptcobj.name
        enabled = ptcobj.enabled
        if ptcobj.id != 0: name = name + '.' + str(ptcobj.id)
        tmpAction = QtGui.QAction(name,self)
        tmpAction.setCheckable(True)
        tmpAction.setChecked(enabled)
        tmpAction.setToolTip(ptcobj.fullfilename)
        self.menuViewPointClouds.addAction(tmpAction)
        self.toolBarBottom.addAction(tmpAction)

    def addViewDisplayVectors(self,name):
        tmpAction = QtGui.QAction(name,self)
        tmpAction.setCheckable(True)
        self.menuViewDisplayVectors.addAction(tmpAction)
        self.toolBarLeft.addAction(tmpAction)
        if self.previousTriggeredViewDisplayVectorsAction == None:
            self.previousTriggeredViewDisplayVectorsAction = tmpAction
        return tmpAction

    def statusMsg(self,msg):
        log.info("Status: "+str(msg))
        self.console.appendPlainText(str(msg))

#-----------------------------------------------------------------------------
    def loadSettings(self):
        self.statusMsg("Settings loaded")
        settings = QtCore.QSettings("drd","concorde")
        if settings.contains("WindowGeometry"):
            geometrySet = False
            for screen in range(QtGui.QApplication.desktop().numScreens()):
                dr = QtGui.QApplication.desktop().screenGeometry(screen)
                wr = settings.value("WindowGeometry").toRect()
                if dr.contains(wr):
                    self.setGeometry(wr)
                    geometrySet = True
                    break
            if not geometrySet:
                log.warn("Can't set window geometry, outside of bounds!")
        if settings.contains("splitterSizes"): self.splitter.restoreState(settings.value("splitterSizes").toByteArray())

#-----------------------------------------------------------------------------
    def saveSettings(self):
        self.statusMsg("Settings saved")
        settings = QtCore.QSettings("drd","concorde")
        settings.setValue("WindowGeometry",QtCore.QVariant(self.geometry()))
        settings.setValue("splitterSizes",QtCore.QVariant(self.splitter.saveState()))

#-----------------------------------------------------------------------------
    def onActionFileNewTriggered(self):
        self.toolBarLeft.clear()
        self.menuViewPointColor.clear()
        self.toolBarBottom.clear()
        self.menuViewPointClouds.clear()
        if self.ptcRenderer != None:
            self.ptcRenderer.clearAll()
        self.glWidget.updateGL()
        self.memoryLabel.setText( formatMemoryStr() )

#-----------------------------------------------------------------------------
    def onActionFileQuitTriggered(self):
        if self.ptcRenderer != None:
            self.ptcRenderer.clearAll()
        self.close()

#-----------------------------------------------------------------------------
    def openFile(self,filename,density,updatecam):
        if self.ptcRenderer == None:
            BEE.initGL()
            self.ptcRenderer = PtcRenderer()
            self.ptcRenderer.framerateLabel = self.framerateLabel

        if len(self.ptcRenderer.ptcobj_array) == 0:
           self.console.clear()

        # first check that this ptc is not already loaded !
        id = 0
        for ptcObj in self.ptcRenderer.ptcobj_array:
            if ptcObj.contains( filename ): id = id +1
            if ptcObj.fullfilename == filename: return

        print "Loading file.. " + filename
        self.statusMsg( "Opened file : " + filename )

        ptcObj = self.ptcRenderer.add_ptc(filename, density, id)
        if ptcObj.density != density:
            self.statusMsg( 'WARNING !! ***** Density forced to ' + str(ptcObj.density) + '% *****' )

        ps = ptcObj.ps
        if len(self.ptcRenderer.ptcobj_array) == 1:
            self.glWidget.setRenderable(self.ptcRenderer,updatecam)
        bounds = ps.get_bounds()

        if len(self.ptcRenderer.ptcobj_array) == 2:  self.actionUseWipe.setEnabled(True)
        else:
            self.hSlider.hide()
            self.vSlider.hide()
            self.actionUseWipe.setEnabled(False)
            self.ptcRenderer.useWipe = False

        if id != 0: self.statusMsg( " - ID = " + str(id) )
        self.statusMsg( " - points count = " + comma_me(str(ps.get_real_point_count())) + " / " + comma_me(str(ps.get_point_count())) )
        self.statusMsg( " - GPU memory used = " + getHumanReadableMemAsStr( ptcObj.gpuSize ) )
        self.statusMsg( " - frameRange = [" + str(ptcObj.min_range) + ' , ' + str(ptcObj.max_range) + ']' )
        self.statusMsg( " - bake width = " + str(ps.get_bake_width()) + ' height = ' + str(ps.get_bake_height()) + ' aspectRatio = ' + str(ps.get_bake_aspect()) )
        self.statusMsg( " - bbox : " )
        self.statusMsg( "   - min = [" + str(bounds.min.x) + ", "+ str(bounds.min.y) + ", " + str(bounds.min.z) + "]")
        self.statusMsg( "   - max = [" + str(bounds.max.x) + ", "+ str(bounds.max.y) + ", " + str(bounds.max.z) + "]")

        self.memoryLabel.setText( formatMemoryStr() )

        # make sure we can see everything...
        if updatecam == True and self.ptcRenderer.keepCameraPositionOnAddingPtc == False :
            #self.onActionCameraUpdateFarClipTriggered()
            #self.glWidget.frameView()
            self.onActionCameraResetTriggered()
            self.onActionCameraUpdateFarClipTriggered()

        self.toolBarLeft.clear()

        self.menuViewPointColor.clear()
        self.addViewPointColor("None").setChecked(True)
        self.addViewPointColor("Position").setChecked(False)
        self.addViewPointColor("Normal").setChecked(False)
        self.addViewPointColor("Radius").setChecked(False)
        for udn in self.ptcRenderer.getUserDataNameList():
            self.addViewPointColor(udn)

        self.toolBarLeft.addSeparator()
        self.menuViewDisplayVectors.clear()
        self.addViewDisplayVectors("None").setChecked(True)
        self.addViewDisplayVectors("Normal").setChecked(False)
        for udn in self.ptcRenderer.getVectorTypeUserDataNameList():
            self.addViewDisplayVectors(udn)

        self.toolBarBottom.clear()
        self.menuViewPointClouds.clear()
        min_range = sys.maxint
        max_range = 0
        for ptcobj in self.ptcRenderer.ptcobj_array:
            self.addViewPointCloud(ptcobj)
            min_range = min(min_range, ptcobj.min_range)
            max_range = max(max_range, ptcobj.max_range)

        if self.frameControlWidget != None:
            self.rightLayout.removeWidget(self.frameControlWidget)
            self.frameControlWidget.close()
            del self.frameControlWidget
        self.frameControlValue = ptcobj.loaded_frame
        self.frameControlWidget = FrameWidget(self.right, 'FrameRange', ptcobj.loaded_frame, {"min":min_range,"max":max_range,"min_hard":min_range,"max_hard":max_range}, self.onFrameControlValueChanged )
        self.rightLayout.addWidget(self.frameControlWidget)

#-----------------------------------------------------------------------------
    def onActionFileAddTriggered(self):
        filename = ""
        path = QtGui.QFileDialog.getOpenFileName(self,"Open file", filename, ("Ptc file (*.ptc)"))
        if path is None or str(path).strip()=="":
            return
        filename = str(path)
        self.openFile(filename, 100, True)

#-----------------------------------------------------------------------------
    def onActionFileCloseTriggered(self):
        ptcinfos_array = self.ptcRenderer.getAllInvisiblePtcInfos()
        self.onActionFileNewTriggered()

        for ptcinfo in ptcinfos_array:
            # fullfilename then density
            self.openFile(ptcinfo[0], ptcinfo[1], False)
            self.glWidget.updateGL()

        if len(self.ptcRenderer.ptcobj_array) == 0: self.console.clear()
        self.memoryLabel.setText( formatMemoryStr() )

#-----------------------------------------------------------------------------
    def onActionFileRefreshTriggered(self):
        ptcinfos_array = self.ptcRenderer.getAllPtcInfos()
        self.onActionFileNewTriggered()
        for ptcinfo in ptcinfos_array:
            # fullfilename then density
            self.openFile(ptcinfo[0], ptcinfo[1], False)
        self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionCameraResetTriggered(self):
        self.glWidget.camera = Camera()
        self.glWidget.frameView()
        self.actionCameraShowShotView.setChecked( False )
        self.actionCameraShowBakeView.setChecked( False )
        self.onActionCameraUpdateFarClipTriggered()

#-----------------------------------------------------------------------------
    def onActionCameraShowShotViewTriggered(self):
        self.actionCameraShowShotView.setChecked( True )
        self.actionCameraShowBakeView.setChecked( False )
        self.onCameraVacFrameSliderValueChanged( int(self.cameraVacFrameControlWidget.get_value()) )

#-----------------------------------------------------------------------------
    def onActionCameraShowBakeViewTriggered(self):
        self.glWidget.camera = Camera()
        self.onActionCameraUpdateFarClipTriggered()
        self.glWidget.camera.pos = self.ptcRenderer.bake_cam_position
        self.glWidget.camera.lookat = self.glWidget.camera.pos + self.ptcRenderer.bake_cam_look_at
        self.glWidget.camera.up = self.ptcRenderer.bake_cam_up
        self.actionCameraShowShotView.setChecked( False )
        self.actionCameraShowBakeView.setChecked( True )
        self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionCameraTargetSelectionTriggered(self):
        self.glWidget.target_selection()
        self.statusMsg( 'New Camera Lookat : ' + str(self.glWidget.camera.lookat) )
        self.glWidget.display_selection_marker = True
        self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionCameraResetExposureTriggered(self):
        self.exposureSlider.setValue(0)

#-----------------------------------------------------------------------------
    def onActionCameraResetDiscPointTransitionDistanceTriggered(self):
        self.discPointTransitionDistanceSlider.setValue(10)

#-----------------------------------------------------------------------------
    def onValueChangedHSlider(self,value):
        if self.ptcRenderer != None:
            self.ptcRenderer.vSliderValue = 0.0
            self.ptcRenderer.hSliderValue = ((value+1) * 0.01) * 2 - 1
            if self.ptcRenderer.hSliderValue == 0.0: self.ptcRenderer.hSliderValue = 0.001
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onValueChangedVSlider(self,value):
        if self.ptcRenderer != None:
            self.ptcRenderer.hSliderValue = 0.0
            self.ptcRenderer.vSliderValue = ((value+1) * 0.01) * 2 - 1
            if self.ptcRenderer.vSliderValue == 0.0: self.ptcRenderer.vSliderValue = 0.001
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onCameraVacFrameSliderValueChanged(self,value):
        value = int(value - self.vacCamInfosArray[ 0 ].frame)

        cameraViewProj = None
        if self.actionCameraShowShotView.isChecked() == True:
            self.glWidget.camera = Camera()
            self.glWidget.camera.view = self.vacCamInfosArray[ value ].matrix
            self.glWidget.camera.proj = grind.util.util.MakeProjection(self.vacCamInfosArray[ value ].fov, self.vacCamInfosArray[ value ].aspect, self.vacCamInfosArray[ value ].near_clip, self.vacCamInfosArray[ value ].far_clip)
            self.glWidget.camera.updateViewProj = False

        if self.ptcRenderer.shot_frustum_display == True:
            view = self.vacCamInfosArray[ value ].matrix
            proj = grind.util.util.MakeProjection(self.vacCamInfosArray[ value ].fov, self.vacCamInfosArray[ value ].aspect, self.vacCamInfosArray[ value ].near_clip, self.vacCamInfosArray[ value ].far_clip)
            view = view.inverse()

            camPos = pimath.V3f( view[3][0], view[3][1], view[3][2] )
            camR = pimath.V3f( view[0][0], view[0][1], view[0][2] )
            camU = pimath.V3f( view[1][0], view[1][1], view[1][2] )
            camA = pimath.V3f( view[2][0], view[2][1], view[2][2] )
            n = self.vacCamInfosArray[ value ].near_clip
            f = self.vacCamInfosArray[ value ].far_clip

            tgt = math.tan( self.vacCamInfosArray[ value ].fov / 2  )
            hn = n * tgt
            wn = hn * self.vacCamInfosArray[ value ].aspect
            hf = f * tgt
            wf = hf * self.vacCamInfosArray[ value ].aspect

            self.ptcRenderer.n0 = camPos + camR * wn + camU * hn - camA * n
            self.ptcRenderer.n1 = camPos - camR * wn + camU * hn - camA * n
            self.ptcRenderer.n2 = camPos + camR * wn - camU * hn - camA * n
            self.ptcRenderer.n3 = camPos - camR * wn - camU * hn - camA * n
            self.ptcRenderer.f0 = camPos + camR * wf + camU * hf - camA * f
            self.ptcRenderer.f1 = camPos - camR * wf + camU * hf - camA * f
            self.ptcRenderer.f2 = camPos + camR * wf - camU * hf - camA * f
            self.ptcRenderer.f3 = camPos - camR * wf - camU * hf - camA * f

        self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionCameraLoadVacFrameInfosTriggered(self):
        filename = ""
        #filename = "/var/tmp/r4567478_vac_61a_030.h5"
        path = QtGui.QFileDialog.getOpenFileName(self,"Open Vac file", filename, ("Vac file (*.h5)"))
        if path is None or str(path).strip()=="": return
        self.loadCameraVacFile( str(path) )

#-----------------------------------------------------------------------------
    def loadCameraVacFile(self, filename):
        print 'Loading Vac File ' + filename
        f = vacpy.File(filename)

        cam = list(f["Hierarchy"].itervalues())[0]
        centerCam = [ i for i in list(cam.itervalues()) if i.nodeType() == "Camera"][0]

        startFrame = cam.getDatablock("__matrix").getDataset("Time")[0][0].item()
        endFrame = cam.getDatablock("__matrix").getDataset("Time")[0][1]

        #self.actionCameraLoadVacFrameInfos.enabled = True
        self.actionCameraShowShotView.setEnabled( True )
        self.actionDisplayShotFrustum.setEnabled( True )

        # add the widget
        if self.cameraVacFrameControlAdded == False:
            self.cameraVacFrameControlWidget = AttribIntWidget(self.right,"VacFrame", startFrame, {"min":startFrame,"max":endFrame,"min_hard":startFrame,"max_hard":endFrame}, self.onCameraVacFrameSliderValueChanged)
            self.rightLayout.addWidget(self.cameraVacFrameControlWidget)
            self.cameraVacFrameControlAdded = True

        horizontal_aperture = centerCam.getDatablock("__horizontalFilmAperture").getDataset("Data")[0].item()
        vertical_aperture = centerCam.getDatablock("__verticalFilmAperture").getDataset("Data")[0].item()
        near_clip = centerCam.getDatablock("@@nearClipPlane").getDataset("Data")[0].item()
        far_clip = centerCam.getDatablock("@@farClipPlane").getDataset("Data")[0].item()

        focal_distance_data = centerCam.getDatablock("__focalDistance").getDataset("Data")
        focal_length_data = centerCam.getDatablock("__focalLength").getDataset("Data")
        matrix_data = cam.getDatablock("__matrix").getDataset("Data")
        num_frames = len(matrix_data)

        self.vacCamInfosArray = []
        for eachFrame in range(0,num_frames):
            VCI = VacCamInfos( startFrame + eachFrame )

            if len(focal_distance_data) == num_frames: VCI.focal_distance = focal_distance_data[eachFrame][0].item()
            else: VCI.focal_distance = focal_distance_data.item()
            if len(focal_length_data) == num_frames: VCI.focal_length = focal_length_data[eachFrame][0].item()
            else: VCI.focal_length = focal_length_data.item()

            VCI.horizontal_aperture = horizontal_aperture
            VCI.vertical_aperture = vertical_aperture
            VCI.aspect = VCI.horizontal_aperture / VCI.vertical_aperture
            VCI.fov = VCI.horizontal_aperture / VCI.focal_length
            VCI.fovy = VCI.vertical_aperture / VCI.focal_length

            VCI.near_clip = near_clip
            VCI.far_clip = far_clip
            matrix = matrix_data[eachFrame]
            VCI.matrix = pimath.M44f( matrix[0][0].item(), matrix[0][1].item(), matrix[0][2].item(), matrix[0][3].item(), \
                                      matrix[1][0].item(), matrix[1][1].item(), matrix[1][2].item(), matrix[1][3].item(), \
                                      matrix[2][0].item(), matrix[2][1].item(), matrix[2][2].item(), matrix[2][3].item(), \
                                      matrix[3][0].item(), matrix[3][1].item(), matrix[3][2].item(), matrix[3][3].item() )
            VCI.matrix = VCI.matrix.inverse()
            self.vacCamInfosArray.append( VCI )

#-----------------------------------------------------------------------------
    def onActionKeepCameraPositionOnAddingPtcTriggered(self):
        if self.ptcRenderer != None:
            self.ptcRenderer.toggleKeepCameraPositionOnAddingPtc()

#-----------------------------------------------------------------------------
    def onActionCameraUpdateFarClipTriggered(self):
        self.glWidget.camera.zmin = 1000
        self.glWidget.camera.zmax = 0.0001
        if self.ptcRenderer != None:
            for ptcObj in self.ptcRenderer.ptcobj_array:
                ps = ptcObj.ps
                bounds = ps.get_bounds()

                minc = pimath.V3f( self.glWidget.camera.pos.x - bounds.min.x,
                                    self.glWidget.camera.pos.y - bounds.min.y,
                                    self.glWidget.camera.pos.z - bounds.min.z )
                maxc = pimath.V3f( self.glWidget.camera.pos.x - bounds.max.x,
                                    self.glWidget.camera.pos.y - bounds.max.y,
                                    self.glWidget.camera.pos.z - bounds.max.z )

                if minc.length() > self.glWidget.camera.zmax:
                    self.glWidget.camera.zmax = minc.length()*100
                    self.glWidget.camera.zmin = 0.0001 * self.glWidget.camera.zmax
                if maxc.length() > self.glWidget.camera.zmax:
                    self.glWidget.camera.zmax = maxc.length()*100
                    self.glWidget.camera.zmin = 0.0001 * self.glWidget.camera.zmax
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionDisplayBBoxTriggered(self):
        if self.ptcRenderer != None:
            self.ptcRenderer.toggle_bbox_display()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionDisplayBakeFrustumTriggered(self):
        if self.ptcRenderer != None:
            self.ptcRenderer.toggle_bake_frustum_display()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionDisplayShotFrustumTriggered(self):
        if self.ptcRenderer != None:
            self.ptcRenderer.toggle_shot_frustum_display()
            self.onCameraVacFrameSliderValueChanged( int(self.cameraVacFrameControlWidget.get_value()) )
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionViewBackFacesTriggered(self):
        if self.ptcRenderer != None:
            self.ptcRenderer.toggle_view_back_faces()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionViewLightingTriggered(self):
        if self.ptcRenderer != None:
            self.ptcRenderer.toggle_lighting()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionUseWipeTriggered(self):
        if self.ptcRenderer != None:
            b = self.ptcRenderer.toggle_useWipe()
            if b: self.hSlider.show(); self.vSlider.show()
            else: self.hSlider.hide(); self.vSlider.hide()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionViewDisplayModePointsTriggered(self):
        self.actionViewDisplayModePoints.setChecked(True)
        self.actionViewDisplayModeDiscs.setChecked(False)
        if self.ptcRenderer != None:
            self.ptcRenderer.use_point_shader()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionViewDisplayModeDiscsTriggered(self):
        self.actionViewDisplayModePoints.setChecked(False)
        self.actionViewDisplayModeDiscs.setChecked(True)
        if self.ptcRenderer != None:
            self.ptcRenderer.use_disc_shader()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionViewIncrPtcTreeDepth(self):
        if self.ptcRenderer != None:
            for ptcObj in self.ptcRenderer.ptcobj_array:
                ptcObj.ps.incr_ptc_tree_bbox_display_targeted_depth()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def onActionViewDecrPtcTreeDepth(self):
        if self.ptcRenderer != None:
            for ptcObj in self.ptcRenderer.ptcobj_array:
                ptcObj.ps.decr_ptc_tree_bbox_display_targeted_depth()
            self.glWidget.updateGL()

#-----------------------------------------------------------------------------
    def closeEvent(self,event):
        #if self.onActionCloseTriggered():
        for p in ps_stack:
            if p[1] != None: del p[1]
        if self.ptcRenderer != None:
            self.ptcRenderer.clearAll()
        ui.ConcordeWindow.closeEvent(self,event)
        self.saveSettings()
        event.accept()
        #else:
        #    event.ignore()

#-----------------------------------------------------------------------------
# stolen from mangle
class AttribFloatWidget(ui.AttribFloatWidget):
    def __init__(self,parent,name,value,extra,callback=None):
        ui.AttribFloatWidget.__init__(self,parent,name,value,extra,callback)
class AttribIntWidget(ui.AttribIntWidget):
    def __init__(self,parent,name,value,extra,callback=None,canUseSlider=True):
        ui.AttribIntWidget.__init__(self,parent,name,value,extra,callback,canUseSlider)
class FrameWidget(ui.FrameWidget):
    def __init__(self,parent,name,value,extra,callback):
        ui.FrameWidget.__init__(self,parent,name,value,extra,callback)

#-----------------------------------------------------------------------------
def main():
    app = QtGui.QApplication(sys.argv)

    if '--help' in sys.argv or '-h' in sys.argv:
        print 'Concorde - version ' + os.getenv('DRD_CONCORDE_VERSION')
        print 'From ' + os.getenv('DRD_CONCORDE_ROOT')
        print 'Please call Stephane Bertout in Output Tech for any bugs or features request.. '
        print
        print 'Simple usage: \n\tconcorde my_file.ptc \n\tconcorde my_file_1.ptc my_file_2.ptc \n\tconcorde *.ptc \n'
        print 'To specify a 50% density for ex: \n\tconcorde -d50 my_file.ptc \n'
        print 'You can also control the density per each file: \n\tconcorde -d50 my_file.ptc -d100 other_file.ptc \n'
        print 'Or as a state: \n\tconcorde -d10 my_huge_file.ptc -d50 other_file_1.ptc other_file_2.ptc \n'
        print 'To specify the slot size of the cache: \n\tconcorde -c4 my_file.1010.ptc '
        print 'This way you can quickly switch between one frame and another (in this case 4 frames will be kept in memory) \n'
        print 'To specify the camera vac file to load: \n\tconcorde -vMyCameraVacFile.h5 my_file.1010.ptc '
        exit(0)

    main_window = ConcordeWindow(sys.argv[1:len(sys.argv)] if len(sys.argv)>1 else None)
    main_window.setWindowTitle( 'Concorde ' + str(os.getenv('DRD_CONCORDE_VERSION')) + ' - Dr. D Studios' )
    main_window.show()
    sys.exit(app.exec_())

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
if __name__=="__main__":
    main()

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

