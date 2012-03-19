#
# Copyright 2010 Dr D Studios Pty Limited (ACN 127 184 954) (Dr. D Studios), its
# affiliates and/or its licensors.
#

from grenade.translators import asset, courier, group, human_user, note, playlist, project, publish_event, reply, scene, sequence, spool, step, shot, task, ticket, version
    
# Supported CLI commands
#
commands = ['create', 'read', 'update', 'delete']

# Entity mappings, to translators and other special info we can't obtain from the associated shotgun schema
#
mappings = {'Asset':{'translator':asset.AssetTranslator},
            'CustomEntity06': {'translator':courier.CourierTranslator},
            'Group':{'translator':group.GroupTranslator},
            'HumanUser':{'translator':human_user.HumanUserTranslator},
            'Note':{'translator':note.NoteTranslator},
            'Playlist':{'translator':playlist.PlaylistTranslator},
            'Project':{'translator':project.ProjectTranslator},
            'PublishEvent':{'translator':publish_event.PublishEventTranslator},
            'Reply':{'translator':reply.ReplyTranslator},
            'Scene':{'translator':scene.SceneTranslator},
            'Sequence':{'translator':sequence.SequenceTranslator},
            'CustomEntity10':{'translator':spool.SpoolTranslator},
            'Step':{'translator':step.StepTranslator},
            'Shot':{'translator':shot.ShotTranslator},
            'Task':{'translator':task.TaskTranslator, 'never_editable': ['dependency_violation', 'time_logs_sum']},
            'Ticket':{'translator':ticket.TicketTranslator, 'never_editable': ['time_logs_sum']},
            'Version':{'translator':version.VersionTranslator}}

# Unsupported API datatypes (for properties, available within shotgun, but not the API)
#
unsupported_api_datatypes = ['summary', 'pivot_column']

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

