CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(ilmbase_COMPONENTS Imath Half)
set(boost_COMPONENTS serialization)
drd_find_packages(PREFIX pkgs AUTO)

FILE(GLOB cpp_files "core/*.cpp" "core/impl/*.cpp" "core/types/*.cpp" "core/util/*.cpp")
FILE(GLOB cpp_serialization_files "core/serialization/*.cpp" )

add_library ( napalm_core_serialization STATIC ${cpp_serialization_files} )
add_library ( napalm_core STATIC ${cpp_files} )
TARGET_LINK_LIBRARIES(napalm_core napalm_core_serialization ${pkgs_LIBRARIES})
INSTALL ( TARGETS napalm_core napalm_core_serialization DESTINATION lib )

# install the headers
FILE(GLOB_RECURSE h_files "*.h")
FILE(GLOB_RECURSE hpp_files "*.hpp")
FILE(GLOB_RECURSE inc_files "*.inc")

# these cpp files are a special case and are treated as include files as well as source
FILE(GLOB_RECURSE inc_cpp_files "core/types/*.inc.cpp")

drd_install_files(
	${h_files} ${hpp_files} ${inc_files} ${inc_cpp_files}
	DESTINATION include/napalm
)
