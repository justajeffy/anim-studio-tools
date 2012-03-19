CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

add_definitions(-Wall -Werror)

#add_definitions(-DVERSION=${NAPALM_VERSION})

subdirs(lib)
#subdirs(pylib)
subdirs(test)
