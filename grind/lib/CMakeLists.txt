CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

drd_find_packages(PREFIX pkgs AUTO)

# CUDA cmake flags must be set AFTER finding cuda
drd_package_in_use("cpuStub" is_cpu)
if(is_cpu)
	set(CUDA_BUILD_EMULATION ON)
endif(is_cpu)

set(CUDA_NVCC_FLAGS "-arch=sm_11")


SET( CPU_CODE
	grind/dump_info.cpp
	grind/context.cpp
	grind/cube.cpp
	grind/type_checks.cpp
	grind/texture.cpp
	grind/program.cpp
	grind/random.cpp
	grind/renderable.cpp
	grind/utils.cpp
)

SET ( GPU_CODE
	grind/cuda_manager.cpp
	grind/mesh.cpp
	grind/mesh.cu
	grind/particle_system.cpp
	grind/line_soup.cpp
	grind/line_soup.cu
	grind/pbo.cpp
	grind/pbo.cu
	grind/device_vector_algorithms.cpp
	grind/device_vector_algorithms.cu
	grind/hash_search.cpp
	grind/hash_search.cu
	grind/bbox.cpp
	grind/vector_set.cpp
	grind/timer.cu
	grind/mesh_subdivide.cpp
	grind/mesh_subdivide.cu
	grind/guide_set.cpp
	grind/guide_set.cu
	grind/utils.cpp

#	grind/ocl_manager.cpp
)

INCLUDE_DIRECTORIES ( . )

CUDA_ADD_LIBRARY ( grind ${CPU_CODE} ${GPU_CODE} STATIC )

TARGET_LINK_LIBRARIES ( grind ${pkgs_LIBRARIES} )


INSTALL ( TARGETS grind DESTINATION lib )



drd_install_cmake(
	DESTINATION 	cmake
	INCLUDE_DIRS	include 	${pkgs_INCLUDE_DIRS}
	LIBRARY_DIRS 	lib 		${pkgs_LIBRARY_DIRS}
	LIBRARIES		grind 		${pkgs_LIBRARIES}
	DEFINITIONS					${pkgs_DEFINITIONS}
)


FILE(GLOB_RECURSE h_files "grind/*.h")
FILE(GLOB_RECURSE cuh_files "grind/*.cuh")
drd_install_files(
	${h_files} ${cuh_files}
	DESTINATION include
)















