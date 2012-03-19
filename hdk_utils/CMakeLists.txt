CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

include(DrdBuild)

drd_find_packages(PREFIX pkgs AUTO)

FILE(GLOB_RECURSE h_files "src/*.h")
drd_install_files( ${h_files} RELATIVE src DESTINATION include/hdk_utils )

FILE(GLOB_RECURSE cpp_files "src/*.cpp")
add_library ( hdk_utils STATIC ${cpp_files} )
target_link_libraries( hdk_utils ${pkgs_LIBRARIES} )

install( TARGETS hdk_utils DESTINATION lib )

drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include 	${pkgs_INCLUDE_DIRS}
	LIBRARY_DIRS 	lib 		${pkgs_LIBRARY_DIRS}
	LIBRARIES		hdk_utils 	${pkgs_LIBRARIES}
	DEFINITIONS					${pkgs_DEFINITIONS}
)
