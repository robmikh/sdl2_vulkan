#pragma once

// STL
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <array>
#include <sstream>

// SDL
#include <SDL.h>
#include <SDL_syswm.h>
#include <SDL_vulkan.h>

// Vulkan
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>

// GLM
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

// Result helpers
inline void check_vulkan(vk::Result result)
{
	if (result != vk::Result::eSuccess)
	{
		throw std::runtime_error("Vulkan call failed!");
	}
}

inline void check_sdl(SDL_bool value)
{
	if (value != SDL_TRUE)
	{
		throw std::runtime_error(SDL_GetError());
	}
}