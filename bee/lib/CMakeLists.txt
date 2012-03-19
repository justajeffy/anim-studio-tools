CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

set(boost_COMPONENTS filesystem thread system)
drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES ( . )

FILE(GLOB_RECURSE cpp_files "*.cpp")
ADD_LIBRARY (bee STATIC ${cpp_files})

target_link_libraries( bee ${pkgs_LIBRARIES} )

install( TARGETS bee DESTINATION lib )

FILE(GLOB_RECURSE h_files "*.h")
drd_install_files(
	${h_files}
	DESTINATION include/bee
)

drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include 	${pkgs_INCLUDE_DIRS}
	LIBRARY_DIRS 	lib 		${pkgs_LIBRARY_DIRS}
	LIBRARIES		bee 		${pkgs_LIBRARIES}
	DEFINITIONS					${pkgs_DEFINITIONS}
)

