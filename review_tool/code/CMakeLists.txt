CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

FILE(GLOB_RECURSE python_files "./*.py")

drd_install_python( py
	FILES
	${python_files}
	DESTINATION .
)

FILE(GLOB_RECURSE some_files "reviewTool/*.ui")

drd_install_files(
	${some_files}
	DESTINATION .
)


FILE(GLOB_RECURSE some_files "reviewTool/*.png")

drd_install_files(
	${some_files}
	DESTINATION .
)


INSTALL(
  PROGRAMS
  "reviewTool/bin/reviewtool"
  DESTINATION
  bin
)

SET (perms ${DRD_FILE_INSTALL_PERMISSIONS})
LIST(APPEND perms OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)


