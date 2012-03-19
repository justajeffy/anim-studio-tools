CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

set(ilmbase_COMPONENTS Imath)
set(boost_COMPONENTS serialization thread)

drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES ( ../lib )

LINK_DIRECTORIES( lib )

CUDA_ADD_EXECUTABLE (napalmCuda_test main.cpp main.cu)

TARGET_LINK_LIBRARIES ( napalmCuda_test napalmCuda ${pkgs_LIBRARIES} )

INSTALL ( TARGETS napalmCuda_test DESTINATION test )
