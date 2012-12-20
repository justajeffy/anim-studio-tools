#!/bin/env python
import operator, sys, os, pickle, pwd, random
import cherrypy
import simplejson as json
import time
import hashlib
import subprocess
import Queue
import logging

from cherrypy.process import plugins
from lib import ajax, template
from tables import *

SHERMAN = "/opt/sherman"

# Set up the queue for the hf5 read requests
h5q = Queue.Queue(10)

class Root(object):
	def __init__(self, data):
		self.data = data

	@cherrypy.expose
	@template.output('index.html')
	def index(self, path='/', filter='type:', category=0, chart=0):
	
		lastupdate = time.strftime("%a, %d %b %Y %H:%M %p", time.localtime(os.path.getmtime("%s/output/sherman.h5" % SHERMAN)))

		return template.render(path=path, filter=filter, category=category, chart=chart, lastupdate=lastupdate)

	@cherrypy.expose
	def tree(self, path, node, filter=''):

		children = []

		# Open h5 file
		h5 = self.openH5()

		# Get the parent
		p = h5.getNode(path)

		# Get the parent objectID
		parent_objectID = p._v_objectID
	
		# Get the MetaData table for filter look-ups
		table = h5.getNode("/", name="MetaData")

		# Cache the metadata for quicker filtering
		if len(filter) > 5: 
				
			# Split the filter
			category, search = filter.split(':')

			if category == 'user':

				# Convert username to uid
				try:
					search = pwd.getpwnam(search)[2]
				except:
					search = int(search)

			metadata = [ x.fetch_all_fields() for x in table.where("(category == '%s') & (name == '%s') & (parent == '%s')" % (category, search, path)) ]

		# Get the list of children for this parent
		nodeList = h5.listNodes(path, classname='Group')

		# Get all children nodes
		for child in nodeList:
		
			# Sparkline data for each child
			sizetrend = ''
			childhistory = self.getRRDhistory(child._v_pathname)
			if len(childhistory) > 1:
				for value in childhistory:
					if not value == 'NaN' and not value == '':
						sizetrend += "%i," % int(float(value))

			# Convert epoch to readable
			hrlmod = time.strftime("%d/%m/%Y %H:%M %p", time.localtime(int(child._v_attrs.modified)))
			hrlacc = ( int(child._v_attrs.accessed) ) > 0 and time.strftime("%d/%m/%Y %H:%M %p", time.localtime(int(child._v_attrs.accessed))) or ""

			# Check if node has any children 
			leaf = ( child._v_nchildren < 1 ) and True or False

			# Check for any filter action
			if len(filter) > 5:						
			
				for data in metadata:
					if child._v_pathname in data:
	
						size   = int(data[4])
						nfiles = int(data[3])
	
						children.append({ 'objectID': child._v_objectID,
										  'id': child._v_pathname,
										  'name': child._v_name,
										  'path': child._v_pathname,
										  'size': int(size),
										  'nfiles': int(nfiles),
										  'lmod': child._v_attrs.modified,
										  'lacc': child._v_attrs.accessed,
										  'hrlmod': hrlmod,
										  'hrlacc': hrlacc,
										  'trend': '<span class="sparkline_%i">%s</span>' % (child._v_objectID, sizetrend[:-1]),
										  'leaf': leaf,
										  'uiProvider':'col',
										  'cls':'master-task',
										  'iconCls':'task-folder' })
			else:
				size   = child._v_attrs.size
				nfiles = child._v_attrs.nfiles
				
				children.append({ 'objectID': child._v_objectID,
								  'id': child._v_pathname,
								  'name': child._v_name,
								  'path': child._v_pathname,
								  'size': int(size),
								  'nfiles': int(nfiles),
								  'lmod': int(child._v_attrs.modified),
								  'lacc': int(child._v_attrs.accessed),
								  'hrlmod': hrlmod,
								  'hrlacc': hrlacc,
								  'trend': '<span class="sparkline_%i">%s</span>' % (child._v_objectID, sizetrend[:-1]),
								  'leaf': leaf,
								  'uiProvider':'col',
								  'cls':'master-task',
								  'iconCls':'task-folder' })			

		self.closeH5(h5)

		cherrypy.response.headers['Content-Type'] = 'application/json'
		return json.dumps(children)

	@cherrypy.expose
	def category(self, category, path):

		h5 = self.openH5()

		table = h5.getNode("/", name="MetaData")

		# Generate appropriate output
		data = []
		for cat in [ x.fetch_all_fields() for x in table.where("(category == '%s') & (path == '%s')" % (category, path)) ]:
		
			size = int(cat[4])
			
			if category == 'user':
				try:
					data.append({ 'name': pwd.getpwuid(int(cat[2]))[0], 'nfiles': int(cat[3]), 'size': size })
				except:
					data.append({ 'name': cat[2], 'nfiles': int(cat[3]), 'size': size })
			elif category == 'type':
				data.append({ 'name': cat[2], 'nfiles': int(cat[3]), 'size': size })
			
		self.closeH5(h5)

		cherrypy.response.headers['Content-Type'] = 'application/json'
		return json.dumps(data)

	@cherrypy.expose
	def graph(self, path, width=600, height=100, start=None, end=None):

		graph  = hashlib.sha224(path).hexdigest()
		subdir = graph[:2]
		rrd    = "%s/rrd/%s/%s.rrd" % (SHERMAN, subdir, graph)

		drange = 1814400
		if start and end:
			start = int(time.mktime(time.strptime(start, "%d/%m/%Y")))
			end   = int(time.mktime(time.strptime(end, "%d/%m/%Y")))
			now   = int(time.mktime(time.strptime(time.strftime("%d/%m/%Y", time.localtime()), "%d/%m/%Y")))

			drange = end - start
			end    = now - end

			if end == 0: end = 'now'

		# Decide whether to use LAST or AVERAGE
		# If the range is more than a month then show AVERAGE, else LAST
		cf = 'AVERAGE' if drange > 2592000 else 'LAST'

		command = [ "rrdtool", "graph", "-",
					"-F", "-E",
					"--start", str(start),
					"--end", str(end),
					"--width", str(width),
					"--height", str(height),
					"--color", "SHADEA#FFFFFF",
					"--color", "SHADEB#FFFFFF",
					"--vertical-label", "Bytes",
					"--base", "1024",
					"--title", path,
					"--font", "DEFAULT:0:Verdana",
					"--lower-limit", "0",
					"DEF:sizeArea=" + rrd + ":size:%s" % cf, "AREA:sizeArea#98a8b988",
					"DEF:size=" + rrd + ":size:%s" % cf, "LINE2:size#ff1800:size",
					"GPRINT:size:LAST: cur\\: %6.2lf%s",
					"GPRINT:size:MIN: min\\: %6.2lf%s",
					"GPRINT:size:AVERAGE: avg\\: %6.2lf%s",
					"GPRINT:size:MAX: max\\: %6.2lf%s\\j"
				  ]

		graph = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]

		cherrypy.response.headers['Content-Type'] = 'image/png'
		return graph

	@cherrypy.expose
	def pie(self, path):

		children = []

		totalSize = 0
		other     = 0
		data      = []

		# Open h5 file
		h5 = self.openH5()

		# Get all children nodes
		for child in h5.listNodes(path, classname='Group'):

			children.append({ 'label': child._v_name, 'data': int(child._v_attrs.size) })
			totalSize += int(child._v_attrs.size)

		self.closeH5(h5)

		# Get the percentages here and combine low percentages to save on JS rendertime
		for child in children:
			percent = 0
			if totalSize > 0:	
				percent = float(child['data']) / totalSize  * 100

			if int(percent) < 4:
				other += int(child['data'])
			else:
				data.append(child)

		# Add other to data
		if other > 0:
			data.append({ 'label': 'other', 'data': other })

		cherrypy.response.headers['Content-Type'] = 'application/json'	
		return json.dumps(data)

	def convert(self, size):

		s = ''
		for l in ('KB', 'MB', 'GB', 'TB'):
			if size > 1024:
				size /= 1024.0
				s = "%3.2f %s" % (size, l)

		return s

	def getRRDhistory(self, node):

		rrd    = hashlib.sha224(node).hexdigest()
		subdir = rrd[:2]

		# Call the shell script to get the last 5 values
		values = subprocess.Popen(["%s/bin/getRRDhistory.sh" % SHERMAN, "%s/%s.rrd" % (subdir, rrd)], stdout=subprocess.PIPE).communicate()[0]

		return values.split('\n')

	# Get an open file from the queue or open new one
	def openH5(self):

		try:
			h5 = h5q.get(False)
		except Queue.Empty:
			try:
				h5 = openFile("%s/output/sherman.h5" % SHERMAN, "r")
			except:
				raise
		
		return h5

	# Keep on queue or close it
	def closeH5(self, h5):

		try:
			h5q.put(h5, False)
		except Queue.Full:
			h5.close()

def main(filename):
	data = {}

	# Global configuration
	cherrypy.config.update({
		'tools.encode.on': True, 'tools.encode.encoding': 'utf-8',
		'tools.decode.on': True,
		'tools.trailing_slash.on': True,
		'tools.staticdir.root': os.path.abspath(os.path.dirname(__file__)),
		'server.socket_host': '0.0.0.0',
		'server.socket_port': 80,
	})

	cherrypy.quickstart(Root(data), '/', {
		'/media': {
			'tools.staticdir.on': True,
			'tools.staticdir.dir': 'static'
		}
	})

if __name__ == '__main__':
	main(sys.argv[1])
