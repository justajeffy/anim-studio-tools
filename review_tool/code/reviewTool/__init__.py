# this determine if tank virtual file path will be used, instead of file system location
# this tool has been burn once too many by the slowness that is tank vfs
# so until it's proven it's competence again, this flag will be set to True
TANK_VFS_DISABLE = True

# since migration over to drd-config, the environment variable DRD_JOB is no longer set per tool
# review tool depends on a project and can not run on it's own.
# We'll the project out of tank, we expect it to look like "/drd/jobs/hf2/tank"
import re, os

if os.environ.get("DRD_JOB")==None:
    if os.environ.get("TANK_PROJECT"):
        os.environ["DRD_JOB"] = os.environ.get("TANK_PROJECT").lower()

ASSET_RENDER_TYPES = ["Character", "Prop", "Stage", "Environment","Skydome"]
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

