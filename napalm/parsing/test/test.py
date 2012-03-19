import napalm.core as nc
import napalm.parsing as np


print("testing napalm v" + nc.getAPIVersion() + " (parsing)...")

# create napalm table
t = nc.ObjectTable()
t[1] = "one"
t["two"] = 2
t["tbl"] = nc.ObjectTable()
t["tbl"][55] = 66
t["tbl"]["hey"] = "ho"

# parse equivalent python dict
s = "{1:'one', 'two':2, 'tbl':{55:66, 'hey':'ho'}}"
t_pydict = np.parseDict(s)

# parse equivalent xml
# TODO this isn't useful until we have full collapse supported

# test for equality
assert(nc.areEqual(t, t_pydict))

# test for equality after clone
t_ = t.clone()
assert(nc.areEqual(t_, t_pydict))

# test for inequality after entry addition
t_ = t.clone()
t_[2] = "extra_entry"
assert(not nc.areEqual(t_, t_pydict))

# test for inequality after entry deletion
t_ = t.clone()
del t_[1]
assert(not nc.areEqual(t_, t_pydict))

# test for inequality after entry alteration
t_ = t.clone()
t_[1] = "two"
assert(not nc.areEqual(t_, t_pydict))

print "success"











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

