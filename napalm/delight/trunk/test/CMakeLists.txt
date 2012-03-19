CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES(../lib)

#------------------------------
# GxProc
#------------------------------

ADD_LIBRARY ( GxProc SHARED GxProc.cpp )
TARGET_LINK_LIBRARIES( GxProc napalmDelight ${pkgs_LIBRARIES} )
SET_TARGET_PROPERTIES ( GxProc PROPERTIES PREFIX "")
INSTALL ( TARGETS GxProc DESTINATION test )

#------------------------------
# GxSxProc
#------------------------------

ADD_LIBRARY ( GxSxProc SHARED GxSxProc.cpp )
TARGET_LINK_LIBRARIES( GxSxProc napalmDelight ${pkgs_LIBRARIES} )
SET_TARGET_PROPERTIES ( GxSxProc PROPERTIES PREFIX "")
INSTALL ( TARGETS GxSxProc DESTINATION test )

#------------------------------
# sxTest
#------------------------------

ADD_EXECUTABLE( sxTest sxTest.cpp )
TARGET_LINK_LIBRARIES( sxTest napalmDelight ${pkgs_LIBRARIES} )
INSTALL ( TARGETS sxTest DESTINATION test )
