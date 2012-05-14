import tank
import time
import pwd


def print_info(path, loc):
	
	if path.__class__ == tank.local.path.RevisionPath:
		# its a revision
		r = path.get_revision()

		# unix timestamp
		mod_time = r.get_modification_date()		
		mod_time_unix = int(time.mktime(mod_time.timetuple()))

		# try tank as default uid
		created_by = r.created_by.get_value()
		uid = 0
		try:
			uid = pwd.getpwnam("tank")[2]
		except:
			pass
			
		if created_by is not None:
			# user(foo) --> "foo"
			created_by_name = created_by.get_name()
			# "foo" --> uid 1234
			try:
				uid = pwd.getpwnam(created_by_name)[2]
			except:
				# let unresolved usernames be uid=0
				pass
		
		# figure out file size
		size = 0
		# is it in our location?
		for l in r.get_locations():
			if l["storage"].get_name() == loc:
				size = r.get_content_size() * l["is_available"]
		
		# print out
		print("%d\t%d\t%s\t%s" % (size, uid, mod_time_unix, path.get_full_path()))

def print_info_r(path, loc):
	path_obj = tank.local.Tm().get(path)
	print_info(path_obj, loc)
	for x in path_obj.get_children():
		print_info_r(x, loc)


print_info_r("/", "default")
