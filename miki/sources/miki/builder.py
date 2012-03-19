#
# Dr. D Studios
# ---------------------------------------------------
"""
Handle building the documentation
"""

import os
import re
import sys
import errno
import shutil
import subprocess
import cStringIO as StringIO

from sphinx import main as sphinx_main
from rodin import logging, terminal

from . import sphinx_config
from . import structure
from . import errors

VALID_TARGETS = ["html", "pdf", "latex", "doxygen", "changes"]

def build(source, destination, defaults=None, targets=None, verbose=False, doxyfile=None, clean=False):
    """
    Build documentation in source outputting to destination directory.
    
    :param source: directory - the docs directory containing conf.py
    :param destination: directory to build in (e.g. _build)
    :param defaults: dictionary of default values to pass to Sphinx configuration (e.g. project name).
    :param targets: list of output targets to build (html, pdf, latex etc). If None a default set is used.  
    :param verbose: if True will output more information as build proceeds
    :param doxyfile: if set will attempt to build doxygen documentation sources first using doxyfile
    :param clean: if True will remove each targets output directory before build
    :raises BuilderError: on fail
    
    .. note::
        
        Sub-folders per documentation type will be made under the destination directory.
            
    .. versionchanged:: 1.0
       Miki now reports all warnings and errors regardless of whether verbose argument is True or False.
        
    """
    log = logging.get_logger()
    important_message_reg = re.compile("(WARNING|ERROR|FAILED)")
    
    # Sphinx likes elements to be interpreted as relative to conf.py so change directory for build to source
    # wrap in a try,finally block to ensure directory gets changed back even if there are errors
    current_directory = os.getcwd()
    os.chdir(source)
    try:
        config_file = structure.get_config_file(source)
        if not os.path.isfile(config_file):
            raise errors.BuilderError("Cannot build - required config file not found at expected location: %s" % config_file)
    
        # update configuration with passed arguments
        if defaults:
            log.info("Adding defaults to configuration: %s" % defaults)
            sphinx_config.__dict__.update(defaults)
            sphinx_config.__dict__.update(sphinx_config.compute_derived_values(**defaults))
    
        # build targets
        if targets is None:
            targets = ["html", "pdf"]
        if doxyfile:
            targets.insert(0, "doxygen")
        
                
        for target in targets:
            if target not in VALID_TARGETS:
                raise errors.BuilderError("Invalid target '%s' specified - must be one of %s." % (target, ", ".join(VALID_TARGETS)))
            
            output = os.path.join(destination, target)
            
            if clean and os.path.exists(output):
                log.info("Cleaning existing output for %s target %s" % (target, output))
                shutil.rmtree(output)
                
            log.info("Building %s in %s" % (target, output))
            try:
                os.makedirs(output)
            except OSError, e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise
    
            # At present the pdf builder breaks the standard code-block directive
            # so it is added in on a need only basis
            if target == "pdf":
                sphinx_config.extensions.append("rst2pdf.pdfbuilder")
            else:
                if "rst2pdf.pdfbuilder" in sphinx_config.extensions:
                    sphinx_config.extensions.remove("rst2pdf.pdfbuilder")
            
            if target == "doxygen":
                # read doxyfile contents
                fd = open(doxyfile, "r")
                contents = fd.read()
                fd.close()
                
                # doxygen will take the last specified argument as the main one when encountering the same argument
                # more than once, so can just append the overrides
                contents = contents.split("\n")
                contents.append("PROJECT_NAME = %s" % defaults.get("project", "Unknown"))
                contents.append("OUTPUT_DIRECTORY = %s" % output)
                contents.append("GENERATE_XML = YES")
                contents.append("XML_OUTPUT = xml")
                contents.append("CREATE_SUBDIRS = NO")
                contents = "\n".join(contents)
                
                # now run doxygen in a subprocess
                p = subprocess.Popen(["doxygen", "-"], 
                                     stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     cwd=os.path.dirname(doxyfile))
                output = p.communicate(input=contents)
                if verbose:
                    print output[0]
                if p.returncode != 0:
                    raise errors.BuilderError("Doxygen build failed: %s" % output[0])
            else:
                # Sphinx build
                args = ["sphinx-build"]
                args.extend(["-b", target])
                args.append(source)
                args.append(output)
                
                # redirect output if not verbose
                saved_out = sys.stdout
                saved_err = sys.stderr
                logging.silence("rst2pdf")
                stdout = StringIO.StringIO()
                stderr = stdout
                sys.stdout = stdout
                sys.stderr = stderr
                    
                try:
                    result = sphinx_main(args)
                except Exception, error:
                    pass 
                finally:
                    sys.stdout = saved_out
                    sys.stderr = saved_err
                    
                output = stdout.getvalue()
                
                # parse output for errors and warnings
                failed = False
                if "Exception" in output:
                    log.exception(output)
                else:
                    lines = output.split(os.linesep)
                    for line in lines:
                        match = important_message_reg.search(line)
                        if match:
                            if match.group(1) == 'WARNING':
                                log.warning(line)
                            elif match.group(1) in ('ERROR', 'FAILED'):
                                log.error(line)
                                failed = True
                        elif verbose:
                            log.info(line)
                
                # handle errors
                if failed:
                    raise errors.BuilderError("Errors occurred during build. Use -l/--loud as build argument for full output.")
            
    finally:
        # change directory back
        os.chdir(current_directory)

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

