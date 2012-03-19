
cmake_minimum_required(VERSION 2.6)

project(drd_color)

include(DrdBuild)

# check if we are building in side the source dir
#string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" in_source)
#if(in_source)
#  message(FATAL_ERROR "Requires an out of source build. Please create a separate build directory and run 'cmake path_to_source [options]' in there.")
#endif()

#set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/share/cmake")

# add src/ build target
#add_subdirectory(src)

# add share install target
install(DIRECTORY share DESTINATION ${CMAKE_INSTALL_PREFIX}
        PATTERN ".svn" EXCLUDE
        PATTERN "*~"   EXCLUDE)

# add share install target
install(DIRECTORY src DESTINATION ${CMAKE_INSTALL_PREFIX}
        PATTERN ".svn" EXCLUDE
        PATTERN "*~"   EXCLUDE)

#install(FILES drd_color.xml DESTINATION ${CMAKE_INSTALL_PREFIX})

# add post_install target
add_custom_target(post_install
                  COMMAND chmod -cR ugo-w ${CMAKE_INSTALL_PREFIX}
                  COMMENT "Setting install permissions")
