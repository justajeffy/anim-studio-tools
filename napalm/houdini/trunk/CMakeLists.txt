CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

# check if we are building in side the source dir
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" in_source)
if(in_source)
  message(FATAL_ERROR "Requires an out of source build. Please create a separate build directory and run 'cmake path_to_source [options]' in there.")
endif()

# Perform a python compilation to .pyc
FILE(GLOB_RECURSE python_files "python/*.py")

drd_install_python( py
	FILES
	${python_files}
	DESTINATION .
)

install(DIRECTORY
    houdini
    DESTINATION .
    PATTERN .svn EXCLUDE
)
