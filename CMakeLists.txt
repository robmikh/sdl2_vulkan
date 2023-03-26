cmake_minimum_required(VERSION 3.21)
project(sdl2_vulkan)

if(UNIX AND NOT APPLE)
    set(LINUX 1)
else()
    set(LINUX 0)
endif()

if (NOT APPLE)
    set(APPLE 0)
endif()

add_definitions(
    -DLINUX=${LINUX}
    -DAPPLE=${APPLE}
)

add_definitions(-DNOMINMAX -DSDL_MAIN_HANDLED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_subdirectory(sdl2_vulkan)
