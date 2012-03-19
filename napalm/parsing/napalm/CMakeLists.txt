CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

drd_find_packages(PREFIX pkgs AUTO)

INCLUDE_DIRECTORIES (../../core)
LINK_DIRECTORIES ( lib )

FILE(GLOB_RECURSE cpp_files "*.cpp")

add_library ( napalm_parsing STATIC ${cpp_files} )
TARGET_LINK_LIBRARIES(napalm_parsing napalm_core ${pkgs_LIBRARIES})
INSTALL ( TARGETS napalm_parsing DESTINATION lib )

# install the headers
FILE(GLOB_RECURSE h_files "*.h")

drd_install_files(
	${h_files}
	DESTINATION include/napalm
)
