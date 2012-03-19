import sys
import napalm.core as n
import pimath as p


# globals
fileTypes = [ "nap", "xml" ]
delayLoadableFileTypes = [ "nap" ]


# helper function which returns a table containing all types that napalm uses
def createUberTable():
    t = n.ObjectTable()
    t[1] = 1                    # this is interpreted as an int
    t[2] = 2.2                  # this is interpreted as a float
    t[3] = n.Double(3.333)
    t[4] = "hello"
    t[5] = n.Char(-10)          # in napalm, a char is interpreted as a number
    t[6] = n.UChar(200)         # as is an unsigned char
    t[7] = n.Short(-30000)
    t[8] = n.UShort(60000)
    t[9] = n.UInt(2000000000)

    # todo add the new imath types inc half
    t[10] = p.V3f()
    t[11] = p.V3d()
    t[12] = p.M33f()
    t[13] = p.M33d()
    t[14] = p.M44f()
    t[15] = p.M44d()

    sz = 100
    t[16] = n.CharBuffer(sz)
    t[17] = n.FloatBuffer(sz)
    t[18] = n.DoubleBuffer(sz)
    t[19] = n.IntBuffer(sz)
    t[20] = n.V3fBuffer(sz)
    t[21] = n.V3dBuffer(sz)
    t[22] = n.M33fBuffer(sz)
    t[23] = n.M33dBuffer(sz)
    t[24] = n.M44fBuffer(sz)
    t[25] = n.M44dBuffer(sz)

    t[16].attribs["a1"] = "aaa1"
    t["hey"] = "ho"

    return t


# check types
def testTypes():
    t = createUberTable()
    assert(type(t[1]) == int)
    assert(type(t[2]) == float)
    assert(type(t[3]) == float) # python has just the one real-number type
    assert(type(t[4]) == str)
    assert(type(t[5]) == int)


# basic file tests
def testFile():

    # test1
    fail = False
    try:
        b = n.load("some_nonexistant_file.foo")
    except n.NapalmFileError:
        fail = True
    assert(fail)

    # test2
    f = open("/tmp/notnapalm.txt", "w")
    f.write("this is not a napalm file.")
    f.close()
    fail = False
    try:
        b = n.load("/tmp/notnapalm.txt")
    except n.NapalmSerializeError:
        fail = True
    assert(fail)


# basic buffer tests
def testBuf():

    # test1
    b = n.IntBuffer()
    assert(len(b) == 0)

    # test2
    b = n.IntBuffer(10, 7)
    assert(b[3] == 7)

    # test3
    b = n.IntBuffer(3,2)
    assert(b.contents == [2,2,2])

    # test4
    b = n.IntBuffer(4,6)
    b2 = b.clone()
    assert(b2.contents == [6,6,6,6])
    b2[0] = 7
    assert(b2.contents == [7,6,6,6])


    # test5
    b = n.IntBuffer(10)
    assert(len(b) == 10)
    b[9] = 100
    assert(b[9] == 100)

    fail = False;
    try:
        b[10]
    except IndexError:
        fail = True
    assert(fail)


# basic table tests
def testTable():
    t = n.ObjectTable()
    assert(len(t) == 0)

    fail = False;
    try:
        t["foo"]
    except KeyError:
        fail = True
    assert(fail)

    t[1] = 1.111
    t["two"] = 68
    t[3] = p.V3f(5,6,7)
    t["four"] = 'Q'

    assert(len(t) == 4)
    assert("two" in t)
    assert("blah" not in t)
    assert(t["two"] == 68)
    assert(t["four"] == 'Q')
    vdiff = t[3] - p.V3f(5,6,7)
    assert(vdiff.length() < 0.0001)

    t["buf"] = n.IntBuffer(100)
    t["buf"][50] = 50
    assert(t["buf"][50] == 50)

    del t["four"]
    del t[3]
    assert(len(t) == 3)
    assert(3 not in t)

    keys = t.keys()
    keys2 = []
    for k in t:
        keys2.append(k)
    assert(keys == keys2)


# save an intBuf, load and make sure it's the same
def testIntBufSerialize(ext):
    # create test data on disk
    filepath = "/tmp/intbuf." + ext
    sz = 50
    b = n.IntBuffer(sz)
    b[5] = 5
    n.save(b, filepath)
    b2 = n.load(filepath)

    if ext in delayLoadableFileTypes:
        assert(b2.clientSize() == 0)
    else:
        assert(b2.clientSize() == sz)

    assert(type(b) == type(b2))
    assert(len(b) == len(b2))
    assert(b2[5] == b[5])

    if ext in delayLoadableFileTypes:
        assert(b2.clientSize() == sz)


# save a V3fBuf, load and make sure it's the same
def testV3fBufSerialize(ext):
    # create test data on disk
    filepath = "/tmp/v3fbuf." + ext
    b = n.V3fBuffer(100)
    b[50] = p.V3f(3.3, 4.4, 5.5)
    n.save(b, filepath)

    b2 = n.load(filepath)
    assert(type(b) == type(b2))
    assert(len(b) == len(b2))
    vdiff = b[50] - b2[50]
    assert(vdiff.length() < 0.001)


# save a table, load and make sure it's the same
def testTableSerialize(ext):
    # create test data on disk
    filepath = "/tmp/tbl1." + ext
    t = createUberTable()
    n.save(t, filepath)

    for delayload in [True,False]:
        t2 = n.load(filepath, delayload)
        assert(n.areEqual(t,t2))


# brute-force serialisation testing. Data is serialised via every different combination
# of file type and save/load option, and then checked for equivalence.
def testSerializeBruteForce():
    fileprefix = "/tmp/tblbrute."
    t = createUberTable()

    # save
    files = []
    for ext in fileTypes:
        for compression in [0,1,2]:
            filepath = fileprefix + str(compression) + '.' + ext
            n.save(t, filepath, compression)
            files.append(filepath)

    # load
    for delayload in [True,False]:
        for f in files:
            t2 = n.load(f, delayload)
            assert(n.areEqual(t,t2))


# basic buffer cloning tests
def testBufCloning():
    sz = 10
    b = n.IntBuffer(sz)
    b[4] = 40
    b2 = b.clone()
    assert(b.hasSharedStore(b2))
    assert(not b.uniqueStore())
    assert(not b2.uniqueStore())
    assert(b.storeUseCount() == 2)
    assert(b2.storeUseCount() == 2)

    b3 = b2.clone()
    assert(b3.hasSharedStore(b))
    assert(not b.uniqueStore())
    assert(not b2.uniqueStore())
    assert(not b3.uniqueStore())
    assert(b.storeUseCount() == 3)
    assert(b2.storeUseCount() == 3)
    assert(b3.storeUseCount() == 3)

    b[6] = 66 # will cause copy-on-write
    assert(b.uniqueStore())
    assert(b2.hasSharedStore(b3))
    assert(not b2.uniqueStore())
    assert(not b3.uniqueStore())
    assert(b.storeUseCount() == 1)
    assert(b2.storeUseCount() == 2)
    assert(b3.storeUseCount() == 2)

    b3[2] = 22 # will cause copy-on-write
    assert(b2.uniqueStore())
    assert(b3.uniqueStore())
    assert(not b2.hasSharedStore(b3))


# basic table cloning tests
def testTableCloning():
    t = n.ObjectTable()
    t[1] = 1
    t["two"] = "222"
    t[3] = n.IntBuffer(100)
    t[4] = t[3]
    t[5] = t[3].clone()
    assert(t[3].storeUseCount() == 2)

    t2 = t.clone()
    assert(t.keys() == t2.keys())
    assert(t2[1] == 1)
    assert(t2["two"] == "222")
    assert(t2[3] != t[3])
    assert(t2[3].hasSharedStore(t[3]))
    assert(t2[4] == t2[3])
    assert(t[3].storeUseCount() == 4)


# test delay-loading
def testDelayLoad(ext):
    # create test data on disk
    filepath = "/tmp/dl1." + ext
    sz = 100
    t_ = n.ObjectTable()
    for i in range(10):
        t_[i] = n.IntBuffer(sz)
    n.save(t_, filepath)

    # test1: make sure a buffer's data isn't made resident until it's accessed
    t = n.load(filepath)
    expected_count = sz
    for i in t.iteritems():
        i[1][0] # force resident via zeroeth element read
        count = 0
        for j in t.iteritems():
            count += j[1].clientSize()
        assert(count == expected_count)
        expected_count += sz


# test delay-load + cloning cases (where cloning happens pre-load)
def testDelayLoadAndPreCloning(ext):
    # create test data on disk
    filepath = "/tmp/clpre." + ext
    sz = 13
    t_ = n.ObjectTable()
    b_ = n.IntBuffer(sz)
    b2_ = b_.clone()
    t_[1] = b_
    t_[2] = b2_
    n.save(t_, filepath)

    # test1: Make sure that when buffers are loaded, their cloned relationships are kept intact
    t = n.load(filepath)
    assert(t.keys() == t_.keys())
    assert(t[1].hasSharedStore(t[2]))
    t[2][0] # force resident via zeroeth element read
    assert(t[1].clientSize() == sz)
    assert(t[1].hasSharedStore(t[2]))


# test delay-load + cloning cases (where cloning happens post-load)
def testDelayLoadAndPostCloning(ext):
    # create test data on disk
    filepath = "/tmp/clpost." + ext
    sz = 1000
    b_ = n.IntBuffer(sz)
    b_[60] = 600
    assert(b_.clientSize() == sz)
    n.save(b_, filepath)

    # test1: Make sure that cloning a buffer doesn't force any data resident. Also make
    # sure that if either buffer forces data resident via a read, both buffers get the
    # same resident data.
    b = n.load(filepath)
    assert(b.clientSize() == 0)
    b2 = b.clone()
    assert(b.clientSize() == 0)
    assert(b2.clientSize() == 0)

    assert(b[60] == 600) # will cause buffer to load into mem
    assert(b.clientSize() == sz)
    assert(b2.clientSize() == sz)
    assert(b.hasSharedStore(b2))

    # test2: Make sure that if a buffer's data is non-resident, and then it is cloned,
    # then the clone is accessed for writing, that the original buffer's data is NOT
    # made resident.
    c = n.load(filepath)
    assert(c.uniqueStore())
    assert(c.clientSize() == 0)

    c2 = c.clone()
    assert(c2.clientSize() == 0)
    c2[6] = 66 # will cause data to load for c2, but still not for c...!
    assert(c.clientSize() == 0)
    assert(c2.clientSize() == sz)

def testToString():
    t = n.ObjectTable()
    assert( str(t) == "{}" )
    t[0] = '1'
    t['1'] = 0.0
    t[2] = p.V3f(0,1,2)
    assert( str(t) == '{"1": 0.f, 0: "1", 2: V3f(0, 1, 2)}' )

def testToPyString():
    t = n.ObjectTable()
    assert( t.pyStr()== "{}" )
    t[0] = '1'
    t['1'] = 0.0
    t[2] = p.V3f(0,1,2)
    assert( t.pyStr() == '{"1": 0., 0: "1", 2: V3f(0, 1, 2)}' )

def testToTupleString():
    t = n.ObjectTable()
    assert( t.tupStr() == "{}" )
    t[0] = '1'
    t['1'] = 0.0
    t[2] = p.V3f(0,1,2)
    assert( t.tupStr() == '{"1": 0., 0: "1", 2: (0, 1, 2)}' )
    exec("m="+t.tupStr())
    assert( m == {"1":0.,0:"1",2:(0,1,2)})


########### main

print("testing napalm v" + n.getAPIVersion() + " (core)...")

assert(n.getTotalClientBytes() == 0)

testTypes()

testFile()
testBuf()
testTable()
testBufCloning()
testTableCloning()
testSerializeBruteForce()

for ft in fileTypes:
    testIntBufSerialize(ft)
    testV3fBufSerialize(ft)
    testTableSerialize(ft)

for ft in delayLoadableFileTypes:
    testDelayLoad(ft)
    testDelayLoadAndPreCloning(ft)
    testDelayLoadAndPostCloning(ft)

assert(n.getTotalClientBytes() == 0)

testToString()
testToPyString()
testToTupleString()

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

