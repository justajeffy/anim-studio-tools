##
#   \namespace  reviewTool.database
#
#   \remarks    Defines the main database class that will be used
#               for the Review Tool
#   
#   \author     Dr. D Studios
#   \date       07/27/11
#

from .. import settings

from PyQt4.QtGui import QMessageBox

class MissingSession(object):
    def find( self, *args, **kwds ):
        return []
    
    def find_one( self, *args, **kwds ):
        return None

class Database(object):
    def __init__( self ):
        self._session   = None
        self._cache     = {}
        self._project   = None
    
    def cache( self, key, value ):
        self._cache[str(key)] = value
    
    def cachedValue( self, key, default = None ):
        return self._cache.get(str(key),default)
    
    def clearCache( self ):
        self._cache.clear()
    
    def defaultShotFields( self ):
        return [ 'id',
                    'code',
                    'sg_cut_order',
                    'sg_handle_start',
                    'sg_handle_end'
                    'sg_cut_start',
                    'sg_cut_end',
                    'sg_status_list'
        ]
            
    
    def findDepartment( self, name ):
        return self.session().find_one( settings.SHOTGUN_DEPARTMENT_TYPE, [['code','is',name]] )
    
    def findScene( self, name ):
        # if the lookup is a dictionary already, just pass it back
        if ( type(name) == dict ):
            return name
        
        # lookup the scene based on its name
        cache_key = 'scene::%s' % name
        
        if ( not cache_key in self._cache ):
            filters = []
            filters.append(['sg_status_list','is_not','omt'])
            filters.append(['code','is',name])
            filters.append(['project','is',self.project()])
            
            scene = self.session().find_one('Scene',filters)
            self._cache[cache_key] = scene
            return scene
        else:
            return self._cache[cache_key]
    
    def findScenes( self ):
        # retrieve the cached value
        cache_key = 'scenes'
        
        if ( not cache_key in self._cache ):
            # create the search filters
            filters = []
            filters.append(['sg_status_list','is_not','omt'])
            filters.append(['project','is',self.project()])
            
            # create the fields request
            fields  = ['id','code','sg_sort_order']
            scenes  = self.session().find('Scene',filters,fields)
            self._cache[cache_key] = scenes
            
            return scenes
        else:
            return self._cache[cache_key]
    
    def findShot( self, code, fields = None ):
        """
            Looks up the shot based on the inputed shot code
            
            :param      code:
            :type       <str>:
            
            :param      fields:
            :type       <list> [ <str>, .. ] || None:
            
            :return     <dict>:
        """
        cache_key = 'shot::%s' % code
        shot = self._cache.get(cache_key)
        if ( not shot ):
            if ( not fields ):
                fields = self.defaultShotFields()
            
            shot = self.session().find_one( 'Shot', [['code','is',str(code)]],fields)
            self._cache[cache_key] = shot
        return shot
    
    def findShotSiblings( self, shot, fields = None, padLeft = 1, padRight = 1 ):
        """
            Looks up the siblings for the inputed shot based on the given padding
            
            :param      shot:
            :type       <str> || <dict>:
            
            :param      padLeft:
            :type       <int>:
            
            :param      padRight:
            :type       <int>:
            
            :return     <tuple> (<list> [ <dict>, .. ] before,<list> [ <dict>, .. ] after)
        """
        # make sure we have some padding, otherwise, just return a blank list
        if ( not (padLeft or padRight) ):
            return ([],[])
            
        # make sure we have valid shot information
        if ( type(shot) != dict ):
            shot = self.session().find_one( 'Shot', [['code','is',str(shot)]],['sg_cut_order','code'] )
        elif ( not ('sg_cut_order' in shot and 'code' in dict) ):
            # make sure we have a valid condition
            if ( not 'id' in shot ):
                raise Exception('Invalid cut information supplied')
                
            shot = self.session().find_one( 'Shot', [['id','is',shot['id']]],['sg_cut_order','code'] )
        
        if ( not fields ):
            fields = self.defaultShotFields()
        
        # lookup the sibling shots based on the inputed data
        scene_code      = shot['code'].split('_')[0]
        sg_scene        = self.session().find_one( 'Scene', [['code','is',scene_code]] )
        base_cut_order  = shot['sg_cut_order']
        
        # lookup previous siblings
        previous = []
        if ( padLeft ):
            prev_filter = []
            prev_filter.append(['sg_scene','is',sg_scene])
            prev_filter.append(['sg_cut_order','less_than',base_cut_order])
            prev_filter.append(['sg_status_list','is_not','omt'])
            
            prev_order = [ { 'field_name': 'sg_cut_order', 'direction':'desc' } ]
            
            previous = self.session().find( 'Shot', 
                                            prev_filter,
                                            fields,
                                            prev_order,
                                            limit = padLeft)
            previous.reverse()
            
        # lookup following siblings
        following = []
        if ( padRight ):
            next_filter = []
            next_filter.append(['sg_scene','is',sg_scene])
            next_filter.append(['sg_cut_order','greater_than',base_cut_order])
            next_filter.append(['sg_status_list','is_not','omt'])
            
            next_order = [ { 'field_name': 'sg_cut_order', 'direction':'asc' } ]
            
            following = self.session().find( 'Shot', 
                                            next_filter,
                                            fields,
                                            next_order,
                                            limit = padRight)
        
        return (previous,following)
    
    def findShotsByScene( self, name, fields = None ):
        # retrieve cached value
        cache_key = 'scene_shots::%s' % name
        
        # if no cache is found, then create a new query
        if ( not cache_key in self._cache ):
            # create the search filters
            filters = []
            filters.append(['sg_status_list','is_not','omt'])
            filters.append(['sg_scene','is', self.findScene(name)])
            filters.append(['project','is',self.project()])
            
            if ( not fields ):
                fields = self.defaultShotFields()
            
            shots   = self.session().find('Shot',filters,fields)
            self._cache[cache_key] = shots
            return shots
        else:
            return self._cache[cache_key]
     
    def project( self ):
        if ( self._project == None ):
            self._project = self.session().find_one('Project',[['sg_short_name','is',settings.PROJECT_NAME]])
        return self._project
    
    def session( self ):
        # initialize the session
        if ( self._session == None ):
            from shotgun_v3 import Shotgun
            try:
                self._session = Shotgun(settings.SHOTGUN_SERVER,settings.SHOTGUN_USER,settings.SHOTGUN_KEY,settings.SHOTGUN_VERSION)
            except:
                QMessageBox.critical( None, 'Could not connect to Shotgun', 'Could not establish a connection to the shotgun server: %s\n\nMany of the features of the Review Tool will not work properly.' % settings.SHOTGUN_SERVER )
                self._session = MissingSession()
        
        return self._session
    
    def uncache( self, key ):
        key = str(key)
        if ( key in self._cache ):
            return self._cache.pop(key)
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

