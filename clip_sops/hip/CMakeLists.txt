CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FILE(GLOB_RECURSE hip_files "*.hip")

drd_install_files( ${hip_files} DESTINATION hip )

