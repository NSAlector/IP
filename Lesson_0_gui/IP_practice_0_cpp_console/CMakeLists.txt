CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)
PROJECT (argus_test)

add_definitions( -DMAGICKCORE_QUANTUM_DEPTH=16 )
add_definitions( -DMAGICKCORE_HDRI_ENABLE=0 )
find_package(ImageMagick COMPONENTS Magick++)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)

include_directories(${JPEG_INCLUDE_DIRS})
#message(STATUS "include_dir='${JPEG_INCLUDE_DIRS}'")
#include_directories(${PNG_INCLUDE_DIRS})
#message(STATUS "include_dir='${PNG_INCLUDE_DIRS}'")

include_directories(${ImageMagick_INCLUDE_DIRS})
include_directories(include)

file(GLOB SOURCES_CIP "src/add_src/*.cpp")
file(GLOB SOURCES_EXE "src/main.cpp")

set(cip SOURCES_CIP)
set(argus_test SOURCES_EXE)

add_library(cip SHARED ${SOURCES_CIP})
add_executable(argus_test ${SOURCES_EXE})

target_link_libraries(cip ${ImageMagick_LIBRARIES})
target_link_libraries(cip ${JPEG_LIBRARIES})
target_link_libraries(cip ${PNG_LIBRARIES})

target_link_libraries(argus_test cip)

