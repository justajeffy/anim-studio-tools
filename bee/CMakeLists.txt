CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

include(DrdBuild)

set(freeimage_STATIC ON)

# no can do, glew not built with -fPIC hence can't statlink to pyBee.so
#set(glew_STATIC ON)


subdirs( lib )
subdirs( py )

# for the moment - to build + run sample, uncomment this, and uncomment Qt
# in the package.yaml, then uncomment the export $PATH line in package.yaml,
# then do a drd-build + install. Then drd-env into a bee env, and run 'beesample'
# whilst in the samples subdir.
#subdirs( samples )
