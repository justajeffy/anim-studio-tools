CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)
include(DrdInstallMikiDocs)

drd_install_dirs(
    sources/
    DESTINATION ./sources/
)

drd_install_dirs(runtimes DESTINATION ${CMAKE_INSTALL_PREFIX} EXECUTABLE)
drd_install_miki_docs(tank_hf2)

SET (perms ${DRD_FILE_INSTALL_PERMISSIONS})
LIST(APPEND perms OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE)