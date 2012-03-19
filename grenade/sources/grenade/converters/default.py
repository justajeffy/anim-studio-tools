#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from ..common.error import GrenadeConnectionError, GrenadeValidationError

"""
Default set of translation converters for Grenade.

.. versionchanged:: 1.4.0
    Wrap shotgun API find() calls in a try/except, capture any errors (that may spawn horrible long
    reams of HTML encoded error text) and return our own user/developer-friendly error feedback instead.
"""

def convert_asset(session, asset_name):
    """
    Translate the given asset name into the Shotgun representation for an asset.
    
    :param session:
        An active Shotgun session.
    :param asset_name:
        Name of an asset in Shotgun (e.g., 'mumble'). This value is matched against the Asset entity
        'code' field.
    :returns:
        Shotgun Asset entity dictionary.
        
    .. versionadded:: v00_04_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    try:
        asset = session.find_one('Asset', filters=[['code', 'is', asset_name]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Asset")')
    
    if asset:
        return asset
    else:
        raise GrenadeValidationError('Invalid asset name : %s' % asset_name)

def convert_datetime(session, timestamp):
    """
    Translate the given date/time stamp into a Shotgun API compatible datetime representation.
    
    :param session:
        An active Shotgun session.
    :param timestamp:
        The timestamp to convert, formatted as 'dd/mm/yyyy HH:MM' (e.g., '30/08/2010 15:00').
    :returns:
        Datetime object.
        
    .. versionadded:: v00_05_00
    """
    from datetime import datetime
    
    try:
        return datetime.strptime(timestamp, "%d/%m/%Y %H:%M")
    except ValueError, e:
        raise GrenadeValidationError('Invalid timestamp : %s' % e)
       
def convert_department(session, department_name):
    """
    Translate the given department name into the Shotgun representation for a department.
    
    :param session:
        An active Shotgun session.
    :param asset_name:
        Name of a department in Shotgun (e.g., 'anim'). This value is matched against the 
        CustomNonProjectEntity03 (Department) entity 'code' field.
    :returns:
        Shotgun CustomNonProjectEntity03 (Department) entity dictionary.
        
    .. versionadded:: v00_05_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    try:
        department = session.find_one('CustomNonProjectEntity03', filters=[['code', 'is', department_name]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Department")')
    
    if department:
        return department
    else:
        raise GrenadeValidationError('Invalid department name : %s' % department_name)

def convert_group(session, group_name):
    """
    Translate the given group name into the Shotgun representation for a group.

    :param session:
        An active Shotgun session.
    :param login_name:
        Name of a group in Shotgun (e.g., 'vfx'). This value is matched against the Group 
        entity 'code' field.
    :returns:
        Shotgun Group entity dictionary.
        
    .. versionadded:: v00_03_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    try:
        group = session.find_one('Group', filters=[['code', 'is', group_name]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Group")')
        
    if group:
        return group
    else:
        raise GrenadeValidationError('Invalid group name : %s' % group_name)
    
def convert_imgseqs(session, imgseqs):
    """
    Translate the given image sequences into the respective Shotgun representations for each sequence.

    :param session:
        An active Shotgun session.
    :param imgseqs:
        A list of image sequences in Shotgun (e.g., ['900_036_anim_v001', 'sc_07_srpublish_068']). Each 
        entry in the list is matched against the Version entity 'code' field.
    :returns:
        A list of Shotgun Version entity dictionaries.
        
    .. versionadded:: v00_05_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    results = []
    for imgseq in imgseqs:
        try:
            version = session.find_one('Version', filters=[['code', 'is', imgseq]])
        except ValueError, e:
            raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Version")')

        if version:
            results.append(version)
        else:
            raise GrenadeValidationError('Invalid image sequence name : %s' % imgseq)
        
    return results


def convert_link(session, link):
    """
    Translate the given link into the respective Shotgun representation for the entity.
    
    :param session:
        An active Shotgun session.
    :param link:
        An entity in Shotgun, formatted as {<type>:<filters>}, where <type is a valid Shotgun
        entity name and <filters> is a valid list of Shotgun API filters which may be used to 
        retreive the entity (e.g., {'Scene':[['id', 'is', 1]]}).
    :returns:
        Shotgun entity dictionary.
        
    .. versionadded:: v00_06_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    if len(link.keys()) != 1:
        raise GrenadeValidationError('Invalid link format : %s' % link)
    
    for key in link.keys():
        try:
            entity = session.find_one(key, filters=link[key])
        except ValueError, e:
            raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("%s")' % key)
        
    if entity:
        return entity
    else:
        raise GrenadeValidationError('Invalid link : %s' % link)

def convert_links(session, links):
    """
    Translate the given links into the respective Shotgun representations for each linked entity.

    :param session:
        An active Shotgun session.
    :param links:
        A list of entities in Shotgun, formatted as [{<type>:<filters>}, {...}, ...], where <type> is a
        valid Shotgun entity name and <filters> is a valid list of Shotgun API filters which may be used
        to retrieve the entity (e.g., [{'Scene':[['id', 'is', 1]]}, {'Shot':[['id', 'is', 6174]]}]).
    :returns:
        A list of Shotgun entity dictionaries.
        
    .. versionadded:: v00_03_00
    .. versionchanged:: v00_06_00
        Remove reference to note in validation error messages.
    .. versionchanged:: v00_06_00
        Updated to use convert_link() for each item in the provided links list.
    .. versionchanged:: 0.9.0
        Check to make sure we've received a list of links!
    """
    if type(links) != list:
        raise GrenadeValidationError('Invalid links format (a list of links is required) : %s' % links)
    
    results = []
    for link in links:
        results.append(convert_link(session, link))
            
    return results

def convert_meetings(session, meetings):
    """
    Translate the given meetings into the respective Shotgun representations for each one.

    :param session:
        An active Shotgun session.
    :param imgseqs:
        A list of meetings in Shotgun (e.g., ['DD VFX Walkthrough 17/06']). Each entry in the list 
        is matched against the CustomEntity01 entity 'code' field.
    :returns:
        A list of Shotgun CustomEntity01 entity dictionaries.
        
    .. versionadded:: v00_05_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    results = []
    for meeting_name in meetings:
        try:
            meeting = session.find_one('CustomEntity01', filters=[['code', 'is', meeting_name]])
        except ValueError, e:
            raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Meeting")')

        if meeting:
            results.append(meeting)
        else:
            raise GrenadeValidationError('Invalid meeting name : %s' % meeting_name)
        
    return results
    
def convert_project(session, project_name):
    """
    Translate the given project name into the Shotgun representation for a project.

    :param session:
        An active Shotgun session.
    :param project_name:
        Name of a project in Shotgun (e.g., 'hf2'). This value is matched against the Project entity 
        'sg_short_name' field.
    :returns:
        Shotgun Project entity dictionary.
        
    .. versionadded:: v00_03_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    try:
        project = session.find_one('Project', filters=[['sg_short_name', 'is', project_name]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Project")')
        
    if project:
        return project
    else:
        raise GrenadeValidationError('Invalid project name : %s' % project_name)

def convert_recipients(session, recipients):
    """
    Translate the given recipients into the respective Shotgun representations for each user or group.

    :param session:
        An active Shotgun session.
    :param recipients:
        A list of users or groups in Shotgun (e.g., ['luke.cole', 'vfx']). If the recipient is a 
        user name, the value is matched against the HumanUser entity 'login' field. If the recipient 
        is a group name, the value is matched against the Group entity 'code' field. The list may 
        contain a mixture of user and group names.
    :returns:
        A list of Shotgun HumanUser or Group entity dictionaries.
        
    .. versionadded:: v00_03_00
    .. versionchanged:: v00_06_00
        Remove reference to note in validation error message.
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    results = []
    for recipient in recipients:
        try:
            group = session.find_one('Group', filters=[['code', 'is', recipient]])
        except ValueError, e:
            raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Group")')
            
        if group:
            results.append(group)
            continue    # we've got a group, no need to try looking up a matching person
        
        try: 
            person = session.find_one('HumanUser', filters=[['login', 'is', recipient]])
        except ValueError, e:
            raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("HumanUser")')

        if person:
            results.append(person)
        else:
            raise GrenadeValidationError('Invalid recipient : %s' % recipient)
        
    return results

def convert_scene(session, scene):
    """
    Translate the given scene name into the Shotgun representation for a scene.

    :param session:
        An active Shotgun session.
    :param scene:
        The scene to retrieve from Shotgun, formatted as <project_name>:<scene_name> (e.g., 'hf2:19d')
        where <project_name> is the short name of the project to search in (matched against 'sg_short_name'
        on the Project entity), and <scene_name> is the name of the scene to retrieve (matched against
        'code' on the Scene entity).
    :returns:
        Shotgun Scene entity dictionary.
        
    .. versionadded:: v00_04_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    bits = scene.split(':')
    if len(bits) != 2:
        raise GrenadeValidationError('Invalid number of scene specification tokens : %s' % scene)
    
    project = convert_project(session, bits[0])     # this will raise an exception for us if the project is unidentified
    
    try:
        entity = session.find_one('Scene', filters=[['code', 'is', bits[1]], ['project', 'is', project]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Scene")')
        
    if entity:
        return entity
    else:
        raise GrenadeValidationError('Invalid scene specification: %s' % scene)

def convert_sequence(session, sequence_name):
    """
    Translate the given sequence name into the Shotgun representation for a sequence.

    :param session:
        An active Shotgun session.
    :param sequence_name:
        Name of a sequence in Shotgun (e.g., '?'). This value is matched against the Sequence entity 
        'code' field.
    :returns:
        Shotgun Sequence entity dictionary.
        
    .. versionadded:: v00_03_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    try:
        sequence = session.find_one('Sequence', filters=[['code', 'is', sequence_name]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Sequence")')
        
    if sequence:
        return sequence
    else:
        raise GrenadeValidationError('Invalid sequence name : %s' % sequence_name) 
    
def convert_shot(session, shot):
    """
    Translate the given shot specification into the Shotgun representation for a shot.

    :param session:
        An active Shotgun session.
    :param shot:
        The shot to retrieve from Shotgun, formatted as <project_name>:<shot_name> (e.g., 'hf2:19d_010')
        where <project_name> is the short name of the project to search in (matched against 'sg_short_name' 
        on the Project entity), and <shot_name> is the name of the shot to retrieve (matched against 'code' 
        on the Shot entity).
    :returns:
        Shotgun Shot entity dictionary.
        
    .. versionadded:: v00_04_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    bits = shot.split(':')
    if len(bits) != 2:
        raise GrenadeValidationError('Invalid number of shot specification tokens : %s' % shot)
    
    project = convert_project(session, bits[0])     # this will raise an exception for us if the project is unidentified
    
    try:
        entity = session.find_one('Shot', filters=[['code', 'is', bits[1]], ['project', 'is', project]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Shot")')
    
    if entity:
        return entity 
    else:
        raise GrenadeValidationError('Invalid shot specification: %s' % shot)

def convert_step(session, step):
    """
    Translate the given pipeline step name into the Shotgun representation for a pipeline step.

    :param session:
        An active Shotgun session.
    :param step:
        Name and type of a pipeline step in Shotgun, formatted as <entity_type>:<step_name> (e.g., 
        'Shot:Stereo') where <entity_type> is the type of pipeline step to search for (matched
        against 'entity_type' on the Step entity), and <step_name> is the name of the step to
        retrieve (matched against 'code' on the Step entity).
    :returns:
        Shotgun Step entity dictionary.
        
    .. versionadded:: v00_04_00
    .. versionchanged:: 1.2.0
        In order to correctly specify a pipeline step, we need to know the associated entity type
        in addition to the step name (multiple steps with the same name might exist for different
        entity types).
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    bits = step.split(':')
    if len(bits) != 2:
        raise GrenadeValidationError('Invalid number of step specification tokens : %s' % step)
    
    try:
        entity = session.find_one('Step', filters=[['entity_type', 'is', bits[0]], ['code', 'is', bits[1]]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("Step")')
        
    if entity:
        return entity 
    else:
        raise GrenadeValidationError('Invalid step specification : %s' % step)
    
def convert_user(session, login_name):
    """
    Translate the given login name into the Shotgun representation for a user.

    :param session:
        An active Shotgun session.
    :param login_name:
        Name of a user in Shotgun (e.g., 'luke.cole'). This value is matched against the HumanUser 
        entity 'login' field.
    :returns:
        Shotgun HumanUser entity dictionary.
        
    .. versionadded:: v00_03_00
    .. versionchanged:: 1.4.0
        Wrap shotgun API find() calls in try except statements, to capture any errors and return 
        developer/user-friendly feedback.
    """
    try:
        user = session.find_one('HumanUser', filters=[['login', 'is', login_name]])
    except ValueError, e:
        raise GrenadeConnectionError('An internal shotgun error was encountered on session.find_one("HumanUser")')
        
    if user:
        return user
    else:
        raise GrenadeValidationError('Invalid user name : %s' % login_name)

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

