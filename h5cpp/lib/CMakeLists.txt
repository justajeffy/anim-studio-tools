CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )


drd_find_packages(PREFIX pkgs AUTO)

add_definitions(-Wall -Werror)


FILE(GLOB_RECURSE cpp_files "*.cpp")
add_library ( h5cpp SHARED ${cpp_files} )
target_link_libraries( h5cpp ${pkgs_LIBRARIES} )
install( TARGETS h5cpp DESTINATION lib )


drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include 	${pkgs_INCLUDE_DIRS}
	LIBRARY_DIRS 	lib 		${pkgs_LIBRARY_DIRS}
	LIBRARIES		h5cpp 		${pkgs_LIBRARIES}
	DEFINITIONS					${pkgs_DEFINITIONS}
)


FILE(GLOB_RECURSE hpp_files "*.hpp")
FILE(GLOB_RECURSE h_files "*.h")
FILE(GLOB_RECURSE inc_files "*.inc")

drd_install_files( ${h_files} ${hpp_files} ${inc_files} DESTINATION include )
