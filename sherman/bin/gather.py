#!/bin/env python2.5

from __future__ import with_statement
from threading import Thread, activeCount

import re, sys
import os, shutil
import subprocess
import datetime, time
import logging

SHERMAN     = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
PATHFILTER  = SHERMAN + '/conf/paths.conf'
CUSTOMCMDS  = SHERMAN + '/conf/custom.conf'
FINDDIR     = SHERMAN + '/input/'
OUTPUTDIR   = SHERMAN + '/output/'
BINDIR      = SHERMAN + '/bin/'
WEBCONTROL  = SHERMAN + '/www/controller.py'
TIMESTAMP   = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

HDF5LIB     = SHERMAN + "/lib/hdf5/v1_8_5/lib"

# Set up the log file for the find data
logging.basicConfig(filename=FINDDIR + TIMESTAMP, level=logging.INFO, format='%(message)s')

def filters():

	# 0: Path, 1: Path specific rules, 2: Global rules
	regexes = ( re.compile(r"^\+"), re.compile(r"^ \-"), re.compile(r"^\-") )
	
	path = ''
	filters = {'paths': {}, 'global': []}
	with open(PATHFILTER) as pf:
		for line in pf:

			for type, regex in enumerate(regexes):

				# Check if this line contains a path or filter
				try: filter = re.split(regex, line)[1][:-1]
				except: continue

				if type == 0: 
					path = filter
					filters['paths'][path] = []
				if type == 1: 
					filters['paths'][path].append(filter)
				if type == 2:
					filters['global'].append(filter)

	return filters

def customcmds():
	
	# 0: Command to execute
	regexes = ( re.compile("^\+"), )

	cmds = []
	with open(CUSTOMCMDS) as cc:
		for line in cc:

			for type, regex in enumerate(regexes):

				try: cmd = re.split(regex, line)[1][:-1]
				except: continue

				if type == 0: cmds.append(cmd)
	
	return cmds

def find(filters):

	for path, excludes in filters['paths'].iteritems():

		# Create the find command
		cmd = [ "find", path, "-type", "f",
				"-printf", "%s\t%U\t%T@\t%A@\t%p\n" ]

		# Add the excludes - combine specific and global
		for exclude in excludes + filters['global']:
			cmd += [ "-o", "-wholename", exclude, "-prune" ]

		t = Thread( target=logdata, args=(cmd,) )
		t.start()

def execute(cmds):

	for cmd in cmds:

		# Subprocess takes an array of arguments, so split the string here
		t = Thread( target=logdata, args=(cmd.split(),) )
		t.start()

def logdata(cmd):

	print "Processing '%s' at %s" % (' '.join(cmd)[:-1], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

	# Run the command and write the output from all threads to same logfile
	find = subprocess.Popen(cmd, stdout=subprocess.PIPE)

	while 1:
		line = find.stdout.readline()

		if not line: break

		logging.info( line[:-1] )


def main():

	# Run the find commands first
	find(filters())

	# Run any custom commands
	execute(customcmds())

	# Wait for threads to finish
	while activeCount() > 1: time.sleep(1)

	# Generate the hdf5 file
	hd5 = subprocess.Popen(["python2.5", BINDIR + "generate.py", TIMESTAMP], env={"LD_LIBRARY_PATH": HDF5LIB}, stdout=subprocess.PIPE).communicate()[0]

	# Generate/Update the rrd files - process 8 rrdtool cmds at a time using xargs
	cat   = subprocess.Popen(["cat", "/tmp/rrdcommands.tmp"], stdout=subprocess.PIPE)
	xargs = subprocess.Popen(["xargs", "-l", "-P", "8", "rrdtool"], stdin=cat.stdout, stdout=subprocess.PIPE).communicate()[0]

	# Make the tmp hdf5 the active one
	shutil.move(OUTPUTDIR + 'sherman.h5.tmp', OUTPUTDIR + 'sherman.h5')

	# Touch the web controller to force sherman to load the new hdf5
	if os.path.exists(WEBCONTROL): os.utime(WEBCONTROL, None)

	# bzip2 files and keep for later processing - if necessary
	bzip = subprocess.Popen(["bzip2", FINDDIR + TIMESTAMP], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]

if __name__ == '__main__':
	
	main()
