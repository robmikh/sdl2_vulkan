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

// Taken from: https://github.com/KhronosGroup/Vulkan-Hpp/blob/5e8166e2841b59a528c9159791010a1ece298956/samples/utils/utils.cpp#L466-L480
inline uint32_t FindMemoryType(vk::PhysicalDeviceMemoryProperties const& memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask)
{
    uint32_t typeIndex = uint32_t( ~0 );
      for ( uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++ )
      {
        if ( ( typeBits & 1 ) && ( ( memoryProperties.memoryTypes[i].propertyFlags & requirementsMask ) == requirementsMask ) )
        {
          typeIndex = i;
          break;
        }
        typeBits >>= 1;
      }
      assert( typeIndex != uint32_t( ~0 ) );
      return typeIndex;
}