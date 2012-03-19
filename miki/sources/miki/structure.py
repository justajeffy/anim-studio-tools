#
# Dr. D Studios
# ---------------------------------------------------
"""
Manage setup and validation of documentation structure.
"""

import os
import shutil
import errno

from rodin import logging

from . import config
from .errors import StructureError

# Utility functions
# ---------------------------------------------------
def get_docs_directory(project_directory):
    """
    Return docs directory
    """
    return os.path.join(project_directory, "docs")

def get_build_directory(docs_directory):
    """
    Return docs build directory
    """
    return os.path.join(docs_directory, "_build")

def get_config_file(docs_directory):
    """
    Return config file in docs directory
    """
    return os.path.join(docs_directory, "conf.py")


# structure generation and validation
# ---------------------------------------------------
def generate(project_directory, update=False):
    """
    Generate appropriate documentation structure under project_directory
    
    :param project_directory: the project root directory to create docs directory under
    :param update: if True will overwrite existing files

    .. note::
    
        Does not remove files or directories.
    
    """
    log = logging.get_logger()
    destination = get_docs_directory(project_directory)
    source = config.template_path
    
    log.info("Creating structure at %s" % destination)
    if update:
        log.info("Update flag set - will overwrite existing files.")
    _copytree(source, destination, log, overwrite=update)

        
def validate(project_directory):
    """
    Validate structure at project_directory
    
    :param project_directory: the project root directory containing docs directory structure
    
    :returns: True if structure valid else False
    
    """
    log = logging.get_logger()
    log.info("Validating docs structure under %s" % project_directory)

    destination = get_docs_directory(project_directory)
    source = config.template_path
    valid = True
    
    for path, dirs, files in os.walk(source):
        # ignore hidden directories (.svn etc)
        for d in dirs:
            if d.startswith("."):
                dirs.remove(d)
                
        path_parts = path.replace(source, "").split(os.sep)
        path_parts.insert(0, destination)
        dest_path = os.path.join(*path_parts)
        if os.path.isdir(dest_path):
            log.info("Found required directory %s" % (dest_path))
        else:
            valid = False
            log.warn("Missing directory %s" % (dest_path))
            
        for file in files:
            if file.startswith("."):
                continue
            
            file_path = os.path.join(dest_path, file)
            if os.path.isfile(file_path):
                log.info("Found required file %s" % (file_path))
            else:
                valid = False
                log.warn("Missing file %s" % (file_path))
    
    if valid:
        log.info("Structure was valid.")
    else:
        log.warn("Structure was invalid.")
        
    return valid



if not getattr(__builtins__, "WindowsError", None):
    class WindowsError(OSError): pass


def _copytree(src, dst, log, overwrite=False):
    """
    Custom copytree that fills in missing directories and files and doesn't error when encountering existing paths
    
    :param overwrite: if True will overwrite existing files
    """
    names = os.listdir(src)
    try:
        os.makedirs(dst)
    except OSError, why:
        if why.errno == errno.EEXIST:
            log.warn("Directory already existed %s" % dst)
        else:
            raise
    else:
        log.info("Created directory %s" % dst)
             
    for name in names:
        if name.startswith("."):
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        if os.path.isdir(srcname):
            _copytree(srcname, dstname, log, overwrite=overwrite)
        else:
            if not os.path.exists(dstname):
                shutil.copy2(srcname, dstname)
                log.info("Created file %s" % dstname)
            elif overwrite and os.path.exists(dstname):
                shutil.copy2(srcname, dstname)
                log.info("Overwrote existing file %s" % dstname)
            else:
                log.warn("Did not overwrite existing %s" % dstname)
    try:
        shutil.copystat(src, dst)
    except WindowsError:
        # can't copy file access times on Windows
        pass
    except Exception, why:
        raise
    
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

