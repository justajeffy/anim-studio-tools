CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

set(boost_COMPONENTS python)
drd_find_packages(PREFIX pkgs AUTO)

set( beepy_sources
    gl/bindings.cpp
    gl/PyProgram.cpp
    gl/PyTexture.cpp
    gl/PyRenderTarget.cpp
    io/bindings.cpp
    scenegraph/bindings.cpp
    kernel/bindings.cpp
    rootBindings.cpp
)

ADD_DEFINITIONS (-DBOOST_PYTHON_MAX_ARITY=17)

INCLUDE_DIRECTORIES ( . )
INCLUDE_DIRECTORIES ( ../../lib )

LINK_DIRECTORIES ( lib )

ADD_LIBRARY ( pyBee SHARED
    ${beepy_sources}
)

TARGET_LINK_LIBRARIES(pyBee bee ${pkgs_LIBRARIES})

SET_TARGET_PROPERTIES ( pyBee PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES ( pyBee PROPERTIES OUTPUT_NAME "BEE")

INSTALL ( TARGETS pyBee DESTINATION py )
