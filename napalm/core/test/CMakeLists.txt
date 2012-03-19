CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

set(ilmbase_COMPONENTS Imath)
set(boost_COMPONENTS serialization)

drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES ( ../ )

LINK_DIRECTORIES( lib )

ADD_EXECUTABLE (napalm_core_test main.cpp)
ADD_EXECUTABLE (napalm_core_test_spirit spirit.cpp)

TARGET_LINK_LIBRARIES ( napalm_core_test napalm_core ${pkgs_LIBRARIES} )
TARGET_LINK_LIBRARIES ( napalm_core_test_spirit napalm_core ${pkgs_LIBRARIES} )

INSTALL ( TARGETS napalm_core_test DESTINATION test )
INSTALL ( TARGETS napalm_core_test_spirit DESTINATION test )
