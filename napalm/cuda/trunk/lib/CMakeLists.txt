CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(ilmbase_COMPONENTS Imath)
set(boost_COMPONENTS serialization)

drd_find_packages(PREFIX pkgs AUTO)

# minimum architecture required
set(CUDA_NVCC_FLAGS "-arch=sm_13")

FILE(GLOB_RECURSE cpp_files "napalmCuda/*.cpp")
FILE(GLOB_RECURSE cu_files "napalmCuda/*.cu")
cuda_add_library ( napalmCuda SHARED ${cpp_files} ${cu_files} )
TARGET_LINK_LIBRARIES(napalmCuda ${pkgs_LIBRARIES})
INSTALL ( TARGETS napalmCuda DESTINATION lib )

# install the headers
FILE(GLOB_RECURSE h_files "*.h")
drd_install_files( ${h_files} DESTINATION include )

drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include 			${pkgs_INCLUDE_DIRS}
	LIBRARY_DIRS 	lib 				${pkgs_LIBRARY_DIRS}
	LIBRARIES		napalmCuda napalm	${pkgs_LIBRARIES}
	DEFINITIONS							${pkgs_DEFINITIONS}
)
