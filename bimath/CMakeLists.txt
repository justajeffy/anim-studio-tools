CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

include(DrdBuild)

FILE(GLOB_RECURSE hpp_files "src/*.hpp")

drd_install_files(
	${hpp_files}
	RELATIVE src
	DESTINATION include/bimath
)

drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include
)
