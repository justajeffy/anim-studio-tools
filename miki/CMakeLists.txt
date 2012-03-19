CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

FILE(GLOB_RECURSE python_files "sources/*.py")

drd_install_python( py
	FILES
	${python_files}
	DESTINATION .
)

install_files_(runtimes/miki DESTINATION . EXECUTABLE)
install_files_(resources/template/technical/index.rst resources/template/user/index.rst resources/template/conf.py resources/template/index.rst DESTINATION . )
install(DIRECTORY resources/template/static DESTINATION resources/template PATTERN ".svn" EXCLUDE)
