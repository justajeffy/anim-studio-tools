CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

SET(QT_USE_QTOPENGL TRUE)
SET(QT_USE_QTUITOOLS TRUE)

set(boost_COMPONENTS filesystem thread)
drd_find_packages(PREFIX pkgs AUTO)

set(project_SRCS
    src/glSettings.cpp
    src/genericgl.cpp
    src/window.cpp
    src/deferredLightingTest.cpp
    src/forwardLightingTest.cpp
    src/horizonBasedSSAO.cpp
    src/simpleHwShadowTest.cpp
    src/simpleRenderTargetTest.cpp
    src/simpleToonShaderTest.cpp
)

SET(project_MOC_HDRS
    src/glSettings.h
    src/genericgl.h
    src/window.h
    src/deferredLightingTest.h
    src/forwardLightingTest.h
    src/horizonBasedSSAO.h
    src/simpleHwShadowTest.h
    src/simpleRenderTargetTest.h
    src/simpleToonShaderTest.h
)

SET(project_UIS
  src/settings.ui
)

QT4_WRAP_UI(project_SRCS ${project_UIS})
QT4_WRAP_CPP(project_SRCS ${project_MOC_HDRS})

INCLUDE_DIRECTORIES ( src )
INCLUDE_DIRECTORIES ( ../lib )

LINK_DIRECTORIES ( lib )

ADD_EXECUTABLE ( beesample
    ${project_SRCS}
    ${project_MOC_HDRS}
    src/main.cpp
)

TARGET_LINK_LIBRARIES ( beesample bee ${pkgs_LIBRARIES} )

install( TARGETS beesample DESTINATION bin )

install(DIRECTORY
	data glsl
	DESTINATION bin
	PATTERN .svn EXCLUDE
)
















