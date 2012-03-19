CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)
include(DrdInstallMikiDocs)

FILE(GLOB_RECURSE python_files "sources/*.py")

drd_install_python( py
	FILES
	${python_files}
	DESTINATION .
)

drd_install_files(runtimes/grenade_cl.py DESTINATION .)   
drd_install_files(runtimes/grenade_cl DESTINATION . EXECUTABLE)

drd_install_miki_docs()