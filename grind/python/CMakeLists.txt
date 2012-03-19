CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FILE(GLOB_RECURSE python_files "*.py")

drd_install_python( py
	FILES ${python_files}
	DESTINATION py/grind
)
