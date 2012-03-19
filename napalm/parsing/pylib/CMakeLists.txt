CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(boost_COMPONENTS python)

drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES (../napalm/parsing)
INCLUDE_DIRECTORIES (../../core)
LINK_DIRECTORIES ( lib )

FILE(GLOB_RECURSE cpp_files "*.cpp")
ADD_LIBRARY ( pyNapalm_parsing SHARED ${cpp_files} )

TARGET_LINK_LIBRARIES(pyNapalm_parsing napalm_parsing napalm_core ${pkgs_LIBRARIES})

SET_TARGET_PROPERTIES ( pyNapalm_parsing PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES ( pyNapalm_parsing PROPERTIES OUTPUT_NAME "_napalm_parsing")

INSTALL ( TARGETS pyNapalm_parsing DESTINATION py/napalm/parsing )



