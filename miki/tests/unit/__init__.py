#
# Dr. D Studios
# ---------------------------------------------------

import os
import shutil
from rodin import logging

def setup():
    """
    Setup fixtures
    """
    log = logging.get_logger()
    log.info("Generating test project fixture...")
    
    test_project = os.path.join(os.path.dirname(__file__), "fixtures", "test_project")
    from miki import structure
    docs_directory = structure.get_docs_directory(test_project)
    build_directory = structure.get_build_directory(docs_directory)
    
    if os.path.exists(build_directory):
        log.warn("Clearing existing build directory at %s" % build_directory)
        shutil.rmtree(build_directory)
    structure.create(test_project, force=True)
    
    
def teardown():
    """
    Teardown fixtures
    """
    pass
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

