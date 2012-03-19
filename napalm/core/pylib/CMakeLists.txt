CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(boost_COMPONENTS python serialization)
set(ilmbase_COMPONENTS Imath)

drd_find_packages(PREFIX pkgs AUTO)

# I don't like this but it's necessary due to boost::python::extract<T>() giving this warning:
# dereferencing type-punned pointer will break strict-aliasing rules
# ..when T is a non-pod type.
add_definitions(-fno-strict-aliasing)

INCLUDE_DIRECTORIES (../napalm/core)
LINK_DIRECTORIES ( lib )

FILE(GLOB_RECURSE cpp_files "*.cpp")
ADD_LIBRARY ( pyNapalm_core SHARED ${cpp_files} )

TARGET_LINK_LIBRARIES(pyNapalm_core napalm_core ${pkgs_LIBRARIES})

SET_TARGET_PROPERTIES ( pyNapalm_core PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES ( pyNapalm_core PROPERTIES OUTPUT_NAME "_napalm_core")

INSTALL ( TARGETS pyNapalm_core DESTINATION py/napalm/core )



