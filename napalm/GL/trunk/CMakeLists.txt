CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

subdirs(lib)
subdirs(pylib)
subdirs(py)

if(NOT CENTRAL)
	subdirs(test)
endif(NOT CENTRAL)
