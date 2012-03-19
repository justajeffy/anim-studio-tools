
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

include(DrdBuild)

set(CMAKE_CXX_FLAGS "-fopenmp")

drd_package_in_use("cpuStub" is_cpu)
if(is_cpu)
	add_definitions("-D__DEVICE_EMULATION__")
endif(is_cpu)

set(nvidia_gpu_sdk_COMPONENTS CUTIL)

subdirs(lib)
subdirs(pylib)
subdirs(python)
