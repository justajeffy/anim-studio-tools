CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

include(DrdBuild)

subdirs(src)
subdirs(test)

drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include 			${pkgs_INCLUDE_DIRS}
	DEFINITIONS		${pkgs_DEFINITIONS}
)