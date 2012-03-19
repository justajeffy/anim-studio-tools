CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

#----------------------------------------------------------

FILE(GLOB_RECURSE python_files "*.py")

drd_install_python( py
	FILES ${python_files}
	DESTINATION .
)

#----------------------------------------------------------

FILE(GLOB_RECURSE ui_files "ui/*.ui")

drd_install_files(
	${ui_files}
	DESTINATION .
)

#----------------------------------------------------------

install(DIRECTORY
    bin
    DESTINATION .
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    PATTERN .svn EXCLUDE
)

install(DIRECTORY
    glsl
    DESTINATION .
    PATTERN .svn EXCLUDE
)


