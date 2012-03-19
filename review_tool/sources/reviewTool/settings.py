##
#   \namespace  reviewTool.settings
#
#   \remarks    Contains the common global settings that will be used throughout
#               the Review Tool utility
#   
#   \author     Dr. D Studios
#   \date       08/08/11
#

#--------------------------------------------------------------------------------

import ConfigParser
import os

from .kernel    import core

# determine information from environment
DEBUG_MODE              = os.environ.get('DRD_DEBUG','0') == '1'
PROJECT_NAME            = os.environ.get('DRD_JOB','hf2')
USER_NAME               = os.environ.get('USER','general')
SHOT_AUDIO_CONTAINER    = os.environ.get('SHOT_AUDIO_CONTAINER','ShotAudio_v2')
SORT_VERSIONS_BY_DATE   = True

MEDIA_MODE              = 'video'
PLAY_MODE               = 'selection'

APP_BROWSER             = 'firefox'

PATH_CONFIG             = '/tmp/reviewTool2.%s.conf' % USER_NAME
PATH_HELPDOCS           = 'http://prodwiki/mediawiki/index.php/RnD:HF2Projects:ReviewTool2'
PATH_TEMP               = os.path.join( os.environ.get('DRD_TEMP','/Local/tmp'), 'reviewTool' )
PATH_ICONCACHE          = os.path.join( PATH_TEMP, 'icon_cache' )
PATH_SHARE              = os.environ.get('DRD_SHARE','/drd/jobs/%s/wip/users/%s' % (PROJECT_NAME,USER_NAME) )

FILETYPE_SEQUENCE       = ['jpg','png','exr','iff']
FILETYPE_MOVIE          = ['mov','mp4']
FILETYPE_AUDIO          = ['wav']

# create shotgun session information
SHOTGUN_SERVER          = os.environ.get('DRD_SG_SERVER',   'http://shotgun-sandbox' if DEBUG_MODE else 'http://shotgun-api' )
SHOTGUN_VERSION         = os.environ.get('DRD_SG_VERSION',  'api2')
SHOTGUN_USER            = os.environ.get('DRD_SG_USER',     'reviewTool' )
SHOTGUN_KEY             = os.environ.get('DRD_SG_KEY',      'b9fc6d4614201bfe62a20e64da83defc2e58892c')
SHOTGUN_DEPARTMENT_TYPE = os.environ.get('DRD_SG_DEPT_TYPE','CustomNonProjectEntity03')

#--------------------------------------------------------------------------------

# define the default temp paths
if ( not os.path.exists( PATH_TEMP ) ):
    os.makedirs( PATH_TEMP )
if ( not os.path.exists( PATH_ICONCACHE ) ):
    os.makedirs( PATH_ICONCACHE )

# make the share path only if the root path already exists
if ( not os.path.exists( PATH_SHARE ) ):
    os.mkdir( PATH_SHARE )

# create ordering data
REVIEW_TYPE_ORDER = [
    'creative',
    'technical',
    ''
]

#--------------------------------------------------------------------------------

# define the department filters with default ordering and visibility
DEPARTMENT_FILTERS = {
    'lens':         { 'name': 'Lensing',        'order': 0,     'enabled': True },
    'anim':         { 'name': 'Animation',      'order': 1,     'enabled': True },
    'light':        { 'name': 'Lighting',       'order': 2,     'enabled': True },
    'crowd':        { 'name': 'Crowd',          'order': 3,     'enabled': True },
    'moedit':       { 'name': 'MoEdit',         'order': 4,     'enabled': True },
    'rnd':          { 'name': 'RnD',            'order': 5,     'enabled': False },
    'flo':          { 'name': 'Final Layout',   'order': 6,     'enabled': False },
    'comp':         { 'name': 'Compositing',    'order': 7,     'enabled': False },
    'edit':         { 'name': 'Editing',        'order': 8,     'enabled': False },
    'fx':           { 'name': 'FX',             'order': 9,     'enabled': False },
    'art':          { 'name': 'Art',            'order': 10,    'enabled': False },
    'model':        { 'name': 'Model',          'order': 11,    'enabled': False },
    'previs':       { 'name': 'Previs',         'order': 12,    'enabled': False },
    'rig':          { 'name': 'Rig',            'order': 13,    'enabled': False },
    'skydome':      { 'name': 'Skydome',        'order': 14,    'enabled': False },
    'surface':      { 'name': 'Surface',        'order': 15,    'enabled': False },
    'visdev':       { 'name': 'Vis Dev',        'order': 16,    'enabled': False },
    'charfx':       { 'name': 'Char FX',        'order': 17,    'enabled': False },
    'mocap':        { 'name': 'Mo Cap',         'order': 18,    'enabled': False },
    'bulkedit':     { 'name': 'Bulk Edit',      'order': 19,    'enabled': False },
    'charfinal':    { 'name': 'Char Finaling',  'order': 20,    'enabled': False },
    'stereo':       { 'name': 'Stereo',         'order': 21,    'enabled': False }
}

SHOT_FILTERS = {
    'ip':           { 'name': 'In Progress',    'order': 0,     'enabled': True,        'icon': 'img/shot_status/ip.png' },
    'edp':          { 'name': 'Edit Prep',      'order': 1,     'enabled': False,       'icon': 'img/shot_status/edp.png' },
    'hld':          { 'name': 'Hold',           'order': 2,     'enabled': True,        'icon': 'img/shot_status/hld.png' },
    'fin':          { 'name': 'Final',          'order': 3,     'enabled': True,        'icon': 'img/shot_status/fin.png' },
}

VERSION_FILTERS = {
    'dap':          { 'name': 'Dir Approved',       'order': 0,     'enabled': True,    'icon': 'img/render_status/dap.png' },
    'pdirev':       { 'name': 'Pending Dir Review', 'order': 1,     'enabled': True,    'icon': 'img/render_status/pdirev.png' },
    'fcomp':        { 'name': 'Fix Complete',       'order': 2,     'enabled': True,    'icon': 'img/render_status/fcomp.png' },
    'fix':          { 'name': 'Fix Required',       'order': 3,     'enabled': True,    'icon': 'img/render_status/fix.png' },
    'techap':       { 'name': 'Tech Approved',      'order': 4,     'enabled': True,    'icon': 'img/render_status/techap.png' },
    'nfr':          { 'name': 'Not For Review',     'order': 5,     'enabled': True,    'icon': 'img/render_status/nfr.png' },
    'apr':          { 'name': 'Approved',           'order': 6,     'enabled': True,    'icon': 'img/render_status/apr.png' },
    'rev':          { 'name': 'Pending Review',     'order': 7,     'enabled': True,    'icon': 'img/render_status/rev.png' },
    'vwd':          { 'name': 'Viewed',             'order': 8,     'enabled': True,    'icon': 'img/render_status/vwd.png' }
}

#--------------------------------------------------------------------------------

def audioFileTypes():
    """
            Returns the different audio file type
            filters based on the global settings
            
            :return     <str>:
    """
    return 'Audio Files (*.%s)' % (' *.'.join(FILETYPE_AUDIO))

def compareDepartment(a,b):
    """
            Compares the two departments based on the
            current department filter ordering information
            
            :return     <int>: 1 || 0 || -1
    """
    aorder = DEPARTMENT_FILTERS.get(a,{}).get('order',100000)
    border = DEPARTMENT_FILTERS.get(b,{}).get('order',100000)
    
    return cmp(aorder,border)
    
def departmentFilters():
    """
            Returns the current department filters
            
            :return     <dict> { <str> key: <dict> { <str>: <variant> value, .. }, .. }:
    """
    return DEPARTMENT_FILTERS

def departmentLabels(depts):
    """
            Returns a list of the user friendly labels for the
            inputed department keys
            
            :return     <list> [ <str>, .. ]:
    """
    return [ DEPARTMENT_FILTERS.get(key,{}).get('name',key) for key in depts ]

def departmentOrder( dept ):
    """
            Returns the order number for the inputed department
            key based on the current department filter ordering
            information
            
            :return     <int>:
    """
    return DEPARTMENT_FILTERS.get(dept,{}).get('order',100000)

def departments():
    """
            Returns a list of the department keys, sorted
            alphabetically
            
            :return     <list> [ <str>, .. ]:
    """
    keys = DEPARTMENT_FILTERS.keys()
    keys.sort()
    return keys

def desktopPath( relpath ):
    """
            Returns the desktop filepath based on the inputed relative path
            
            :param  relapath:
            :type   <str>:
            
            :return <str>:
    """
    return os.path.join(os.path.expanduser('~/Desktop'),relpath)

def enableDepartment( dept, state ):
    """
            Sets the enabled state for the given department to the inputed state
            
            :param      dept:
            :type       <str>:
            
            :param      state:
            :type       <bool>:
            
            :return <bool>:
    """
    dept = str(dept)
    if ( not dept in DEPARTMENT_FILTERS ):
        return False
    
    DEPARTMENT_FILTERS[dept]['enabled'] = state
    return True

def enabledDepartments():
    """
            Returns a list of all the department keys that are currently set to 
            enabled, sorted by the current filter order.
            
            :return     <list> [ <str>, .. ]:
    """
    keys = [ key for key in DEPARTMENT_FILTERS if DEPARTMENT_FILTERS[key]['enabled'] ]
    keys.sort(compareDepartment)
    return keys

def enabledShots():
    keys = [ key for key in SHOT_FILTERS if SHOT_FILTERS[key]['enabled'] ]
    keys.sort()
    return keys

def iconCachePath( relpath ):
    """
            Returns a joining of the inputed relative
            path with the tool's temporary path location
            
            :return     <str>:
    """
    return os.path.join( PATH_ICONCACHE, relpath )
    
def orderedDepartments():
    """
            Returns a list of the department keys, sorted by the
            current filter order.
            
            :return     <list> [ <str>, .. ]
    """
    keys = DEPARTMENT_FILTERS.keys()
    keys.sort(compareDepartment)
    return keys

def restore():
    """
            Restores the current settings from the config file
            
            :return     <dict> { <str> key: <str> value, .. }
    """
    # restore review tool settings
    options = {}
    parser  = ConfigParser.ConfigParser()
    if ( not DEBUG_MODE ):
        try:
            parser.read(PATH_CONFIG)
        except:
            core.warn( 'Could not read the settings from %s' % PATH_CONFIG )
            return options
    else:
        parser.read(PATH_CONFIG)
    
    # restore the saved options
    for section in parser.sections():
        for option in parser.options(section):
            options['%s::%s' % (section,option)] = parser.get(section,option)
    
    # restore settings level options
    filters = { 'DEPARTMENT_FILTER': DEPARTMENT_FILTERS, 'SHOT_FILTER': SHOT_FILTERS, 'VERSION_FILTER': VERSION_FILTERS }
    for key, value in options.items():
        section, option = key.split('::')
        
        if ( section in filters ):
            enabled, order = value.split('|')
            filters[section][option]['enabled'] = enabled == 'True'
            filters[section][option]['order']   = int(order)
        
    # return the restored settings
    return options
    
def save(options =  {}):
    """
            Saves the current settings to the config file
            
            :param      options:
            :type       <dict> { <str> key: <variant> value, .. }:
            
            :return     <bool>: success
    """
    # record settings options
    filters = { 'DEPARTMENT_FILTER': DEPARTMENT_FILTERS, 'SHOT_FILTER': SHOT_FILTERS, 'VERSION_FILTER': VERSION_FILTERS }
    for section, filter in filters.items():
        for option, settings in filter.items():
            options['%s::%s' % (section,option) ] = '%s|%s' % (settings['enabled'],settings['order'])
    
    # save the config file
    parser = ConfigParser.ConfigParser()
    
    # add the items to the parser
    for key, value in options.items():
        section, option = key.split('::')
        if ( not parser.has_section(section) ):
            parser.add_section(section)
        parser.set( section, option, str(value) )
    
    # save the config settings
    f = open(PATH_CONFIG,'w')
    if ( not DEBUG_MODE ):
        try:
            parser.write(f)
        except:
            f.close()
            core.warn( 'Could not save the settings out to %s' % PATH_CONFIG )
            return False
    else:
        parser.write(f)
        
    f.close()
    return True

def sharePath( relpath ):
    """
            Generates a shared path location based on the inputed relative
            path location
            
            :param  relpath:
            :type   <str>:
            
            :return <str>:
    """
    return os.path.join(PATH_SHARE,relpath)

def shotFilters():
    """
            Returns the current shot filters
            
            :return     <dict> { <str> key: <dict> { <str>: <variant> value, .. }, .. }:
    """
    return SHOT_FILTERS

def tempPath( relpath ):
    """
            Returns a joining of the inputed relative
            path with the tool's temporary path location
            
            :return     <str>:
    """
    return os.path.join( PATH_TEMP, relpath )
    
def versionFilters():
    """
            Returns the current version filters
            
            :return     <dict> { <str> key: <dict> { <str>: <variant> value, .. }, .. }:
    """
    return VERSION_FILTERS

def versionOrder( status ):
    """
            Returns the order based on the inputed status
            
            :return     <int>:
    """
    return VERSION_FILTERS.get(status,{}).get('order',10000)

def videoFileTypes():
    """
            Returns the different video file type
            filters based on the global settings
            
            :return     <str>:
    """
    return 'Image Sequence Types (*.%s);;Movie Types (*.%s)' % ( ' *.'.join(FILETYPE_SEQUENCE), ' *.'.join(FILETYPE_MOVIE) )
    

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

