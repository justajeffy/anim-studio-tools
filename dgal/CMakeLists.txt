CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

include(DrdBuild)

FILE(GLOB_RECURSE h_files "src/*.h")
FILE(GLOB_RECURSE hpp_files "src/*.hpp")

drd_install_files(
	${h_files} ${hpp_files}
	RELATIVE src
	DESTINATION include/dgal
)

drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include
)


#if(NOT CENTRAL)
#	subdirs(test)
#endif(NOT CENTRAL)
