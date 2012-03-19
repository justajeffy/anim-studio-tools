#
# Dr. D Studios
# ---------------------------------------------------

import sys
import os
import re

def main(arguments=None):
    """
    Miki main entry point.
    
    :param arguments: if specified denotes list of arguments to use, else retrieves argument list from sys.argv
    
    """
    from rodin import logging
    from rodin.parsers.argument import ArgumentParser
    
    from miki import builder, structure, autodoc, errors
    
    if arguments is None:
        arguments = sys.argv[1:]
    
    log = logging.get_logger()
    return_code = 0
    
    # handle arguments
    parser = ArgumentParser(description="Documentation Assistant")
    
    common = ArgumentParser(add_help=False)
    common.add_argument("-p", "--project-directory", help="Project directory. ", default=os.path.abspath("."))
    common.add_argument("-s", "--source-directory", help="Source directory. Can be relative to project directory, default is 'sources'.", action="append")
    common.add_argument("-l", "--loud", help="Output more information during operations.", default=False, action="store_true")
    
    subparsers = parser.add_subparsers(dest='mode', help="Use mode -h/--help for mode specific help")
        
    build_parser = subparsers.add_parser("build", help="Build documentation output.", parents=[common])
    build_parser.add_argument("-n", "--project-name", help="Project name.", default=None)
    build_parser.add_argument("-r", "--project-release", help="Project release string, such as 2.0.3.", default=None)
    build_parser.add_argument("-v", "--project-version", help="Project version string, such as 2.0.", default=None)
    
    build_parser.add_argument("-d", "--doxyfile", help="Location of doxyfile. Can be relative to project directory.", default="Doxyfile")
    build_parser.add_argument("-c", "--clean", help="Remove existing target output before build.", default=False, action="store_true")
    build_parser.add_argument("-t", "--target", action="append", choices=builder.VALID_TARGETS, help="A target to build. Can be specified multiple times.")
    
    structure_parser = subparsers.add_parser("structure", help="Manage documentation structure.", parents=[common])
    structure_group = structure_parser.add_mutually_exclusive_group(required=True)
    structure_group.add_argument("-c", "--create", help="Create documentation structure from template.", default=False, action="store_true")
    structure_group.add_argument("-u", "--update", help="Update existing structure to match template. Will overwrite, but not remove existing files.", default=False, action="store_true")
    structure_group.add_argument("-v", "--validate", help="Validate existing structure.", default=False, action="store_true")
    
    create_parser = subparsers.add_parser("autodoc", help="Automatically generate reference files for all sources.", parents=[common])
    create_parser.add_argument("-d", "--output-directory", help="Directory to output generated files in. Can be relative to docs directory.", default='technical')
    create_parser.add_argument("-u", "--update", help="Overwrite existing files.", default=False, action="store_true")
    
    args = parser.parse_args(arguments)

    # sort out common arguments
    # are we in docs directory by accident
    if os.path.basename(args.project_directory) ==  "docs":
        log.warn("Project directory appears to be a documentation directory - switching up a level.")
        args.project_directory = os.path.dirname(args.project_directory)

    if not args.source_directory:
        args.source_directory = ["sources"]
    
    log.info("Project directory is: %s" % args.project_directory)
    docs_directory = structure.get_docs_directory(args.project_directory)

    # now specialise
    if args.mode == "structure":
        if args.create:
            structure.generate(args.project_directory, update=False)
        elif args.update:
            structure.generate(args.project_directory, update=True)
        elif args.validate:
            structure.validate(args.project_directory)
            
    elif args.mode == "autodoc":
        output_directory_root = os.path.join(docs_directory, args.output_directory)
        for source_directory in args.source_directory:
            source_directory = os.path.join(args.project_directory, source_directory)
            output_directory = output_directory_root
            if len(args.source_directory) > 1:
                output_directory = os.path.join(output_directory, os.path.dirname(source_directory))
            autodoc.generate(output_directory, source_directory, update=args.update)
    
    elif args.mode == "build":
        
        if not os.path.isdir(docs_directory):
            log.error("Docs directory does not exist - cannot build documentation! (Expected location: %s)" % docs_directory)
            return 1
                
        defaults = {}
        
        # guess project name if not set
        if args.project_name:
            defaults["project"] = args.project_name
        else:
            defaults["project"] = os.path.basename(args.project_directory)
            # Check for release build which has project name one level higher with an intermediate release directory
            match = re.search("v\d\d_\d\d_\d\d", defaults["project"])
            if match:
                defaults["project"] = os.path.basename(os.path.dirname(args.project_directory))

        log.info("Project name is: %s" % defaults["project"])
            
        if args.project_release:
            defaults["release"] = args.project_release
        else:
            # guess
            release_folder = os.path.basename(args.project_directory)
            match = re.search("v\d\d_\d\d_\d\d", release_folder)
            if match:
                defaults["release"] = release_folder[1:]
            
        if args.project_version:
            defaults["version"] = args.project_version
            
        # auto add discovered Python packages to PYTHONPATH
        defaults["modindex_common_prefix"] = []
        for source_directory in args.source_directory:
            source_directory = os.path.join(args.project_directory, source_directory)
                
            if not os.path.isdir(source_directory):
                log.warn("Source path is not a valid directory: %s" % source_directory )
            else:
                log.info("Scanning source directory: %s" % source_directory)
                python_package_paths = []
                doxyfile = None
                for path, dirs, files in os.walk(source_directory, topdown=True):
                    if "__init__.py" in files:
                        python_package_paths.append(path)
                        # don't descend any further
                        del dirs[:]
                            
                log.info("Looking for Python packages...")
                python_package_paths.sort()
                for path in python_package_paths:
                    path, entry = os.path.split(path)
                    defaults["modindex_common_prefix"].append("%s." % entry)
                    if not path in sys.path:
                        sys.path.insert(0, path)
                        log.info("Added source path to PYTHONPATH: %s" % path)
        
        for entry in defaults["modindex_common_prefix"]:
            log.info("Added package name '%s' to common prefixes." % entry)
        
        # Check for doxyfile
        if not "/" in args.doxyfile:
            args.doxyfile = os.path.join(args.project_directory, args.doxyfile)
        args.doxyfile = os.path.abspath(args.doxyfile)
        if os.path.isfile(args.doxyfile):
            log.info("Found Doxyfile: %s" % args.doxyfile)
        else:
            args.doxyfile = None
            
        # set default target to html
        if not args.target:
            args.target = ["html"]
              
        # Build docs
        build_directory = structure.get_build_directory(docs_directory)
        
        try:
            builder.build(docs_directory, 
                          build_directory, 
                          defaults=defaults, 
                          targets=args.target,
                          verbose=args.loud,
                          doxyfile=args.doxyfile,
                          clean=args.clean)
        except errors.BuilderError, error:
            log.error("Build failed: %s" % error)
            return_code = 1
        except Exception, error:
            log.exception("Build failed")
            return_code = 1
        else:
            log.info("Build successful - results can be found in %s" % build_directory) 

    return return_code
    
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

