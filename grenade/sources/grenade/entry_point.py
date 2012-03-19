#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

import os
import subprocess
import sys

def get_environment():
    """
    Used to get the configured environment for the process.
    
    .. versionadded:: v00_01_00
    .. versionchanged:: 1.5.1
        Use the grenade script user instead of the generic rnd user by default.
    
    .. todo::
        See if we can/should use Meme to manage configuration of the runtime.
    """
    
    env = {}
    
    if os.environ.has_key('GRENADE_SG_HOST'):
        env['SG_HOST'] = os.environ['GRENADE_SG_HOST']
    else:
        env['SG_HOST'] = 'http://shotgun-sandbox'
        
    if os.environ.has_key('GRENADE_SG_USER'):
        env['SG_USER'] = os.environ['GRENADE_SG_USER']
    else:
        env['SG_USER'] = 'grenade'
        
    if os.environ.has_key('GRENADE_SG_SKEY'):
        env['SG_SKEY'] = os.environ['GRENADE_SG_SKEY']
    else:
        env['SG_SKEY'] = '543c7512b9e6dbee4544e314ae7aed7e2151e7e2'
        
    return env

def main(arguments=None):
    """
    Main entry point.
    
    .. versionadded:: v00_01_00
    .. versionchanged:: v00_02_00
        Changed to use vastly improved model implementation, added update support,
        added rodin logging, applied general cleanup.
    .. versionchanged:: v00_03_00
        Updated to use Shotgun connection utility class.
    .. versionchanged:: v00_04_00
        Updated to use config file for mappings and commands settings.
    .. versionchanged:: v00_08_01
        Added support for result sorting to read().
    .. versionchanged:: 1.4.0
        Sort the list of entities that get passed into the '-e' argument parser choices for easier reading.
    
    .. todo::
        Consider the use of rodin subparsers for each mode which is supported by the CLI; allows arguments for each
        mode to be specified more explicitly (and help make things more robust).
    .. todo::
        Implement stronger argument format/value validation.
    .. todo::
        Return some kind of status code on successful/unsuccessful completion (?).
    """
    from grenade.common.model import find, FIND_ONE, ModelBuilder
    from grenade.config import commands, mappings
    
    from rodin import logging
    from rodin.parsers.argument import ArgumentParser
    
    if arguments is None:
        arguments = sys.argv[1:]
    
    # parse command line options
    entities = mappings.keys()
    entities.sort() # sort the list of available entities for easy to read command-line usage documentation
    
    parser = ArgumentParser(description='Grenade CLI.')
    
    parser.add_argument('-m', '--mode', help='Command mode', choices=commands, default='read')
    parser.add_argument('-e', '--entity', help='Shotgun entity to operate on.', choices=entities, default=None, required=True)
    parser.add_argument('-a', '--args', help='Command arguments (depends on selected mode)', default='\"\"', required=True)
    parser.add_argument('-v', '--verbose', help='Set verbosity level', choices='01', default=0)
        
    args = parser.parse_args(arguments)
    
    params = args.args
    
    verbose = False
    if int(args.verbose) > 0:
        logging.set_level(logging.INFO)
    else:
        logging.set_level(logging.WARNING)
    
    entity = args.entity
    command = args.mode
    
    # create a grenade logger
    log = logging.get_logger('grenade')

    # create a shotgun session
    from .utils.connection import Connection
    
    env = get_environment()
    
    shotgun = Connection(env['SG_HOST'], env['SG_USER'], env['SG_SKEY'])
    session = shotgun.connect(log)
    
    if not session:
        sys.exit()      # no point going on if we failed to connect!
        
    # go ahead and run the command
    translator = mappings[entity]['translator']

    if command == 'create':       
        try:        
            log.info('Creating %s where %s\n' % (entity, params))
        
            instance = ModelBuilder()(session, entity, translator(session), **eval(params)).create()
            print instance['id']
        except Exception, e:
            log.error(str(e))
    
    if command == 'read':       
        try:
            log.info('Reading %s where %s\n' % (entity, params))
        
            arguments = eval(params)
            if not arguments.has_key('order'):
                arguments['order'] = []
            
            for instance in find(session, entity, arguments['filters'], arguments['order']):
                print instance
        except Exception, e:
            log.error(str(e))
        
    if command == 'update':   
        try:
            changes = eval(params)
            id = changes.pop('id')
        
            log.info('Updating %s using %s\n' % (entity, params))
            
            instance = find(session, entity, [['id', 'is', id]], mode=FIND_ONE)
            instance.translator = translator(session)
        
            for key in changes.keys():
                instance[key] = changes[key]
            
            print instance.update()
        except Exception, e:
            log.error(str(e))
        
    if command == 'delete':
        try:
            log.info('Deleting %s where %s\n' % (entity, params))
            
            for instance in find(session, entity, eval(params)): # could be dangerous, just deletes whatever matches find filter
                print instance.delete()
        except Exception, e:
            log.error(str(e))

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

