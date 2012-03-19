#
# Dr. D Studios
# ---------------------------------------------------
"""
.. versionadded:: 1.0

Utilities for auto documenting project.
"""

import os
import shutil
import errno
import re

from rodin import logging

from . import config
from .errors import AutodocError
from . import structure

class Autodocumenter(object):
    """
    An autodocumenter - for test purposes only
    """
    
    def __init__(self):
        """Init"""
        pass
    
    def get_modules(self):
        """
        Get a list of modules to autodocument
        """
        pass
    
    def set_output(self, destination):
        """
        Set the output directory to destination
        """
        pass
    
     
def generate(output_directory, source_directory, update=False):
    """
    Automatically generate autodoc templates for all files in sources
        
    :param output_directory: documentation directory to create autodoc structure under (e.g. docs/technical)
    :param source_directory: directory containing source files for autodoc parsing
    :param update: if True will overwrite existing files

    .. note::
    
        Does not remove files or directories.
        
    """
    log = logging.get_logger()
    
    if not os.path.isdir(source_directory):
        log.error("Could not autodoc - source path is not a valid directory: %s" % source_directory)
        return False
    
    log.info("Output directory set to: %s" % output_directory)
    log.info("Update existing files set to: %s" % update)

    try:
        os.makedirs(output_directory)
    except Exception:
        if not os.path.isdir(output_directory):
            log.error("Could not autodoc - output directory did not exist and could not be created")
            return False 
    
    log.info("Scanning for source files in %s" % source_directory)

    # recursively generate files
    _generate(source_directory, output_directory, update=update, log=log)
        
        
def generate_module_contents(title, module, automodule_options=None, with_inheritance_diagram=True, with_listing=True):
    """
    Construct and return the text contents for a module page.
    
    :param title: title for the page
    :type title: string
    :param module: the module path (in dotted notation)
    :type module: string
    :param automodule_options: list of options to pass to automodule directive
    :type automodule_options: list
    :param with_inheritance_diagram: include a directive to show an inheritance diagram for the module contents
    :type with_inheritance_diagram: boolean
    :param with_listing: include a directive to auto generate a listing of module code contents
    :type with_listing: boolean
    
    :returns: text contents for file
    :rtype: string
    """
    if automodule_options is None:
        automodule_options = [":members:", ":undoc-members:", ":show-inheritance:"]
    
    text = [""]
    heading = "%s (``%s``)" % (title, module)
    text.append("*"*len(heading))
    text.append(heading)
    text.append("*"*len(heading))
    text.append("")

    if with_listing:
        text.append(".. autolisting::")
        text.append("")
    
    if with_inheritance_diagram:
        text.append(".. inheritance-diagram:: %s" % module)
        text.append("")
    
    text.append(".. automodule:: %s" % module)
    for option in automodule_options:
        text.append("    %s" % option)
    text.append("")
    return os.linesep.join(text)
        
        
def generate_index_contents(title, toc, toc_options=None):
    """
    Construct and return the text contents for an index page.
    
    :param title: the title for the page
    :type module: string
    :param toc: the entries for the table of contents
    :type toc: list of strings
    :param toc_options: list of options to pass to toctree directive
    :type toc_options: list
    
    :returns: text contents for file
    :rtype: string
    """
    if toc_options is None:
        toc_options = [":maxdepth: 1"]
    
    text = [""]
    heading = "%s" % (title)
    text.append("="*len(heading))
    text.append(heading)
    text.append("="*len(heading))
    text.append("")

    text.append(".. toctree::")
    for option in toc_options:
        text.append("    %s" % option)
    text.append("")
    
    for entry in toc:
        text.append("    %s" % entry)
    text.append("")
        
    return os.linesep.join(text)

        
def contains_code(path):
    """
    Return whether the file contains interesting code or not
    """
    code_reg = re.compile("^\s*(def |class )")
    
    fd = open(path, "r")
    for line in fd:
        if code_reg.search(line):
            fd.close()
            return True
    fd.close()
    return False
    
    
def contains_class(path):
    """
    Return whether the file contains a class definition
    """
    code_reg = re.compile("^\s*(class )")
    
    fd = open(path, "r")
    for line in fd:
        if code_reg.search(line):
            fd.close()
            return True
    fd.close()
    return False
    
    
def _generate(source, target, package=None, log=None, update=False):
    """
    Walk the tree top down from source generating necessary documents
    
    Generates:
    * A folder for each folder encountered.
    * A Python module autodoc reference file for each Python module encountered
    * An index file in each auto generated folder that acts as table of contents for folder.
    
    :param source: directory containing source files
    :type source: string
    :param target: directory to place generated files and new folders in
    :type target: string
    :param package: package parts for current Python package
    :type package: list
    :param log: package parts for current Python package
    :type log: logger instance to use for log messages (will generate one if not specified)
    :param update: will overwrite existing files if set to True
    :type update: boolean
    
    """
    if not log:
        log = logging.get_logger()
        
    name = os.path.basename(source)
    ext = ".rst"
    index_name = "index"
    toc = []
    if package is None:
        package = []
    
    # get items
    items = os.listdir(source)
    
    # is this path a package
    if "__init__.py" in items:
        package.append(name)
    else:
        package = []
    
    for item in items:
        
        # ignore certain items (such as hidden ones)
        if item[0] in ("."):
            continue
        
        item_path = os.path.join(source, item)
                
        if os.path.isdir(item_path):
            # continue walking tree
            _generate(item_path, os.path.join(target, item), package=package[:], log=log, update=update)
            
            # add to table of contents as item/index
            toc.append("%s/%s" % (item, index_name))
            
        elif os.path.isfile(item_path):
            # only interested in Python files
            if item_path.endswith(".py"):
                
                # only include __init__.py if it contains interesting code
                if item == "__init__.py" and not contains_code(item_path):
                    log.info("Excluded package file because it contained no code contents: %s" % item_path)
                    continue
                
                # generate contents for module
                package.append(item[:-3])
                module = ".".join(package)
                package.pop()
                
                title = item[:-3]
                if title == "__init__":
                    title = package[-1]
                title = title.replace("_", " ").strip().title()
                
                # is there code in the file - if so include a code listing summary
                with_listing = False
                if contains_code(item_path):
                    with_listing = True
                                    
                # is there a class definition in the file - if so include inheritance diagrams 
                with_inheritance_diagram = False
                if contains_class(item_path):
                    with_inheritance_diagram = True
                    
                contents = generate_module_contents(title, 
                                                    module, 
                                                    with_inheritance_diagram=with_inheritance_diagram,
                                                    with_listing=with_listing)
                                
                # write the file 
                destination = os.path.join(source.replace(source, target), "%s%s" % (item[:-3], ext)) 
                if write_file(destination, contents):
                    log.info("Generated module file %s" % destination)
                elif update:
                    if write_file(destination, contents, overwrite=update):
                        log.info("Updated module file %s" % destination)
                else:
                    log.warn("Path exists - will not overwrite %s" % destination)
                
                # add to table of contents
                toc.append(item[:-3])
    
    # create index file
    toc.sort()
    title = "Contents"
    if package:
        title = " ".join([w.title() for w in package])
    contents = generate_index_contents(title, toc)
    
    # write the file 
    destination = os.path.join(source.replace(source, target), "index%s" % (ext)) 
    if write_file(destination, contents):
        log.info("Generated index file %s" % destination)
    elif update:
        if write_file(destination, contents, overwrite=update):
            log.info("Updated index file %s" % destination)
    else:
        log.warn("Path exists - will not overwrite %s" % destination)
    
        
def write_file(path, contents, overwrite=False):
    """
    Write contents to file. If overwrite is False will not overwrite file if already exists.
    Will also create any necessary directories
    
    :param path: location to write file at
    :type path: string
    :param contents: text contents to write to file
    :type contents: string
    :param overwrite: if True will overwrite existing files
    :type overwrite: boolean
    
    """
    log = logging.get_logger()
    
    if os.path.exists(path) and not overwrite:
        return False
    
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
        log.info("Created missing directory %s" % directory)
        
    fd = open(path, "w")
    fd.write(contents)
    fd.close()
    
    return True
        
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

