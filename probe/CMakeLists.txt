CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)
include(DrdInstallMikiDocs)

FILE(GLOB_RECURSE python_files "sources/*.py")

drd_install_python( py
	FILES
	${python_files}
	DESTINATION .
)

drd_install_files(config/noserc config/pylintrc DESTINATION .)
drd_install_dirs(runtimes DESTINATION ${CMAKE_INSTALL_PREFIX} EXECUTABLE)
        
drd_install_miki_docs()
