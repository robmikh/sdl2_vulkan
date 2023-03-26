#pragma once

struct VulkanSurface
{
public:
    VulkanSurface(const VulkanSurface&) = delete;
    VulkanSurface& operator=(const VulkanSurface&) = delete;
    VulkanSurface& operator=(VulkanSurface&& other) = delete;

    VulkanSurface(vk::Instance instance, SDL_Window* window) : m_instance(instance)
    {
        check_sdl(SDL_Vulkan_CreateSurface(window, instance, &m_vkSurface));
    }

    ~VulkanSurface()
    {
        vkDestroySurfaceKHR(m_instance, m_vkSurface, nullptr);
    }

    VkSurfaceKHR get() { return m_vkSurface; }

private:
    vk::Instance m_instance;
    VkSurfaceKHR m_vkSurface = VK_NULL_HANDLE;
};
