CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

set(ilmbase_COMPONENTS Imath)
drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES ( ../src )

ADD_EXECUTABLE (dgal_test main.cpp)
TARGET_LINK_LIBRARIES ( dgal_test ${pkgs_LIBRARIES} )
INSTALL ( TARGETS dgal_test DESTINATION test )

ADD_EXECUTABLE (kdtree_test kdtree_test.cpp)
TARGET_LINK_LIBRARIES ( kdtree_test ${pkgs_LIBRARIES} )
INSTALL ( TARGETS kdtree_test DESTINATION test )


## maya plug test
#ADD_DEFINITIONS ( -DLINUX -D_BOOL -DREQUIRE_IOSTREAM  ${SSE_FLAGS} )
#add_maya_plugin( spatialQuery
#				 maya_plug/mayaAdaptor.hpp
#				 maya_plug/spatialQueryNode.cpp
#
#				)
#
#SET_TARGET_PROPERTIES(spatialQuery
#	    PROPERTIES
#	    PREFIX  "")
#
#target_link_libraries( spatialQuery ${pkgs_LIBRARIES} -lOpenMayaAnim -lOpenMayaUI )
#
#INSTALL ( TARGETS spatialQuery DESTINATION plugins)


