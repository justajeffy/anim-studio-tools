CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

add_definitions(-Wall -Werror)
add_definitions(-DVERSION=${NAPALM_VERSION})
add_definitions(-DBOOST_PYTHON_MAX_ARITY=17)

subdirs(_py)
subdirs(core)
#subdirs(parsing)

# todo need to add component selection to this macro in drd-config

set(ilmbase_COMPONENTS Imath Half)
set(boost_COMPONENTS serialization thread)
drd_find_packages(PREFIX pkgs AUTO)

drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include 			${pkgs_INCLUDE_DIRS}
	LIBRARY_DIRS 	lib 				${pkgs_LIBRARY_DIRS}
	LIBRARIES
		napalm_core
		napalm_core_serialization
		#napalm_parsing
		${pkgs_LIBRARIES}
	DEFINITIONS		${pkgs_DEFINITIONS}
)
