cmake_minimum_required(VERSION 3.21)

find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(GLM REQUIRED)

add_spirv_embed_library(spirv_shaders vert.vert frag.frag)

add_executable(sdl2_vulkan pch.cpp main.cpp)

set_target_properties(sdl2_vulkan PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON)

target_compile_options(sdl2_vulkan PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-g>
    )

target_include_directories(sdl2_vulkan PUBLIC
	$<BUILD_INTERFACE:${SDL2_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${GLM_INCLUDE_DIR}>)

target_link_libraries(sdl2_vulkan PUBLIC
	spirv_shaders Vulkan::Vulkan ${SDL2_LIBRARY})