from __future__ import with_statement
import sys, os, time
import hashlib
import datetime
import subprocess

# Root directory for Sherman
SHERMAN = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]

# Import pytables
sys.path.append(SHERMAN + "/lib")
from tables import *
from numpy import array

LSIN  = sys.argv[1]
H5OUT = SHERMAN + "/output/sherman.h5.tmp"
INPUT = SHERMAN + "/input/"
FSIN  = INPUT + LSIN

# RRD file directory
RRDIR = SHERMAN + "/rrd"

# File for rrd create/update commands to be process with xargs
XARGS = "/tmp/rrdcommands.tmp"
cr = []; up = []

# Depth for generating rrd files
NLEVELS = 10

# Minimum filesize to store in table, ignore anything below
MINFILESIZE = 500000000

# Print out debug/progress
VERBOSE = True

# Create the meta tables to store path, category, name, size
class MetaData(IsDescription):
	path     = StringCol(400)
	category = StringCol(6)
	name     = StringCol(6)
	size     = Int64Col()
	nfiles   = Int64Col()

def generateTree():

	tree = {}

	# Get number of line to show progress bar
	if VERBOSE:
		nlines = subprocess.Popen(["/usr/bin/wc", "-l", FSIN], stdout=subprocess.PIPE).communicate()[0].split()[0]

		print "Generating Tree from input file ..."

	nline = 0

	# Open file
	with open(FSIN) as fh:
		for line in fh:

			try: size, uid, lmod, lacc, path = line.split("\t")
			except: continue

			# Get the file extension for type
			name, type = os.path.splitext(path[:-1])

			# Split path and get filename
			path = path.split('/')
			file = path.pop()

			# Delete empty first element
			del path[0]

			# Store details in dict - for each level of the path
			for i in range(1, len(path) + 1):
				p = '/%s' % '/'.join(path[:i])
				t = type[1:].lower()

				try:
					tree[p]['general']['size']   += int(size)
					tree[p]['general']['nfiles'] += 1

					if tree[p]['general']['modified'] < int(lmod):
						tree[p]['general']['modified'] = int(lmod)

					if tree[p]['general']['accessed'] < int(lacc):
						tree[p]['general']['accessed'] = int(lacc)
				except:
					tree[p] = { 'general': { 'size': int(size), 'nfiles': 1, 'modified': int(lmod), 'accessed': int(lacc), 'changed': False } }

				# Store file type details
				try:
					tree[p]['type'][t]['size']   += int(size)
					tree[p]['type'][t]['nfiles'] += 1
				except:
					
					try:
						tree[p]['type'][t] = { 'size': int(size), 'nfiles': 1 }
					except:
						tree[p]['type'] = { t: { 'size': int(size), 'nfiles': 1 } }

				# Store user details
				try:
					tree[p]['user'][uid]['size']   += int(size)
					tree[p]['user'][uid]['nfiles'] += 1
				except:
					
					try:
						tree[p]['user'][uid] = { 'size': int(size), 'nfiles': 1 }
					except:
						tree[p]['user'] = { uid: { 'size': int(size), 'nfiles': 1 } }

			# Print progress
			if VERBOSE:

				nline += 1

				# Only print every 1000th element to save time
				if nline % 1000 == 0:
					percent = float(nline)/float(nlines)
					sys.stdout.write(str(nline) + "/" + str(nlines) + " processed. " + str(int(percent * 100)) + "% Done.\r")
					sys.stdout.flush()

	# Define MetaData table columns
	dType = [('path', '|S400'), ('category', '|S10'), ('name', '|S10'), ('nfiles', 'uint32'), ('size', 'uint64'), ('parent', '|S400')]

	# Loop of dict and create array of metadata
	if VERBOSE:
	
		print 
		print "Creating Numpy array"

	metaArr = []
	for path, meta in tree.iteritems():
		for category, data in meta.iteritems():
			for name, values in data.iteritems():

				if category != 'general':

					# Check for min filesize, we don't care about peanuts
					if not category == "user" and values['size'] < MINFILESIZE:
						continue

					# Keep reference to parent folder
					levels = path.split('/')[:-1]
					parent = len(levels) == 1 and '/' or '/'.join(levels)

					metaArr.append((path, category, name, values['nfiles'], values['size'], parent))

	# Sort the tree by the path
	if VERBOSE: print "Sorting Tree"
	tree = sorted(tree)

	# Set up filter for compression
	filters = Filters(complevel=1, complib='zlib', fletcher32=False)

	# Open hdf5 file
	h5f = openFile(H5OUT, mode = "w", title = "Filesystem Tree")

	# Add the metadata table - add a numpy array of all the meta data
	metat = h5f.createTable('/', 'MetaData', array(metaArr, dtype = dType), "Meta Table", filters=filters)

	# Print progress
	if VERBOSE:

		# Number of elements in tree
		nelements = len(tree)
		nelement  = 0

		print "Generating HDF5 file ..."

	# Now write data to file
	counter = 0 # Count number of dirs
	for path in tree:

		# Split the path - to count number of levels and the get path/group
		tmpath  = path[0].split("/")

		# Delete empty first element
		del tmpath[0]

		# Sort out details
		nlevels = len(tmpath)
		node    = '/%s' % '/'.join(tmpath[:-1])
		grp     = tmpath[-1]

		# If nlevels is up to a certain depth then generate/update rrd files
		if nlevels < NLEVELS:
			updaterrds(path)
			counter += 1

		# Create the group for this level/node
		group = h5f.createGroup(node, grp, grp)

		# Add attrs
		group._v_attrs.size     = path[1]['general']['size']
		group._v_attrs.nfiles   = path[1]['general']['nfiles']
		group._v_attrs.modified = path[1]['general']['modified']
		group._v_attrs.accessed = path[1]['general']['accessed']
		group._v_attrs.changed  = path[1]['general']['changed']

		if VERBOSE:

			nelement += 1

			# Only print every 1000th element to save time
			if nelement % 1000 == 0:
				percent = float(nelement)/float(nelements)
				sys.stdout.write(str(nelement) + "/" + str(nelements) + " processed. " + str(int(percent * 100)) + "% Done.\r")
				sys.stdout.flush()

	h5f.flush()
	h5f.close()

	if VERBOSE: 
	
		print 
		print "Write commands to file"

	xargsfile = open(XARGS, 'w')
	xargsfile.writelines(cr)
	xargsfile.writelines(up)
	xargsfile.close()

def sorted(tree):
	items = tree.items()
	items.sort()

	return items

def updaterrds(path):
	now = int( time.mktime(time.strptime(LSIN, '%Y-%m-%d-%H-%M-%S')) )

	# Create unique name by hashing it
	rrdname = hashlib.sha224(path[0]).hexdigest()

	# Take the the first two characters to break up into directories
	subdir  = rrdname[:2]

	# Check if a rrd file exists else create
	if not os.path.exists("%s/%s/%s.rrd" % (RRDIR, subdir, rrdname)):

		# Create subdir - if it doesn't exist yet
		if not os.path.exists("%s/%s" % (RRDIR, subdir)):
			os.mkdir("%s/%s" % (RRDIR, subdir))

		create = "create "
		create += "%s/%s/%s.rrd " % (RRDIR, subdir, rrdname)
		create += "--step "
		create += "21600 "
		create += "--start "
		create += str(now - 31536000)
		create += " DS:size:GAUGE:86400:0:U "
		create += "DS:count:GAUGE:86400:0:U "
		create += "RRA:LAST:0.5:1:240 "
		create += "RRA:AVERAGE:0.5:4:720 "
		create += "RRA:AVERAGE:0.5:21:208\n"
		cr.append(create)

	update = "update "
	update += "--template "
	update += "size "
	update += "%s/%s/%s.rrd " % (RRDIR, subdir, rrdname)
	update += "%s:%s\n" % (now, path[1]['general']['size'])
	up.append(update)

if __name__ == '__main__':
	generateTree()

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

