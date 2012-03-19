CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
set(boost_COMPONENTS ${boost_COMPONENTS} mpi)
set(Boost_USE_STATIC_LIBS        ON)
drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES(../lib)

#------------------------------------------------------------------------------
ADD_EXECUTABLE ( test_mpi test_mpi.cpp )
TARGET_LINK_LIBRARIES( test_mpi ${pkgs_LIBRARIES} )
INSTALL ( TARGETS test_mpi DESTINATION test )

#------------------------------------------------------------------------------
ADD_EXECUTABLE ( test_boost_mpi test_boost_mpi.cpp )
TARGET_LINK_LIBRARIES( test_boost_mpi ${pkgs_LIBRARIES} )
INSTALL ( TARGETS test_boost_mpi DESTINATION test )

#------------------------------------------------------------------------------
ADD_EXECUTABLE ( test_boost_non_blocking test_boost_non_blocking.cpp )
TARGET_LINK_LIBRARIES( test_boost_non_blocking ${pkgs_LIBRARIES} )
INSTALL ( TARGETS test_boost_non_blocking DESTINATION test )

#------------------------------------------------------------------------------
ADD_EXECUTABLE ( test_boost_user_type test_boost_user_type.cpp )
TARGET_LINK_LIBRARIES( test_boost_user_type ${pkgs_LIBRARIES} )
INSTALL ( TARGETS test_boost_user_type DESTINATION test )

#------------------------------------------------------------------------------
ADD_EXECUTABLE ( test_napalm_simple test_napalm_simple.cpp )
TARGET_LINK_LIBRARIES( test_napalm_simple ${pkgs_LIBRARIES} )
INSTALL ( TARGETS test_napalm_simple DESTINATION test )
