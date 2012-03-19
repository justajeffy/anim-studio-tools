CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(boost_COMPONENTS python)
#set(ilmbase_COMPONENTS Imath)

drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES (../lib)

LINK_DIRECTORIES ( lib )

FILE(GLOB_RECURSE cpp_files "*.cpp")
ADD_LIBRARY ( pyNapalmDelight SHARED ${cpp_files} )

TARGET_LINK_LIBRARIES( pyNapalmDelight napalmDelight ${pkgs_LIBRARIES} )

SET_TARGET_PROPERTIES ( pyNapalmDelight PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES ( pyNapalmDelight PROPERTIES OUTPUT_NAME "_napalmDelight" )

INSTALL ( TARGETS pyNapalmDelight DESTINATION py/napalmDelight )

