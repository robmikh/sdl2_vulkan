#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <array>
#include <sstream>
#include <SDL.h>
#include <SDL_syswm.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include "spirv_shaders_embedded_spv.h"
#include <vulkan/vulkan.hpp>

inline void check_vulkan(VkResult result)
{
	if (result != VK_SUCCESS)
	{
		std::stringstream stream;
		stream << "Vulkan call failed with: " << result << std::endl;
		throw std::runtime_error(stream.str());
	}
}

inline void check_sdl(SDL_bool value)
{
	if (value != SDL_TRUE)
	{
		throw std::runtime_error(SDL_GetError());
	}
}

int win_width = 1280;
int win_height = 720;

static std::string AppName    = "SDL2/Vulkan";
static std::string EngineName = "Sample Engine";

int main(int argc, const char **argv) 
{
	SDL_SetHint(SDL_HINT_FRAMEBUFFER_ACCELERATION, "1");
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) 
	{
		std::cerr << "Failed to init SDL: " << SDL_GetError() << "\n";
		return -1;
	}

	SDL_Window* window = SDL_CreateWindow("SDL2 + Vulkan",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, win_width, win_height, SDL_WINDOW_VULKAN);

	// Get the required extensions
	std::vector<const char*> extensionNames;
	{
		uint32_t numRequiredExtensions = 0;
		check_sdl(SDL_Vulkan_GetInstanceExtensions(window, &numRequiredExtensions, nullptr));

		extensionNames = std::vector<const char*>(static_cast<size_t>(numRequiredExtensions), nullptr);

		check_sdl(SDL_Vulkan_GetInstanceExtensions(window, &numRequiredExtensions, extensionNames.data()));

		extensionNames.emplace_back(VK_KHR_SURFACE_EXTENSION_NAME);
		extensionNames.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	}

	// Create the Vulkan instance
	vk::UniqueInstance instance;
	{
		uint32_t extensionCount = static_cast<uint32_t>(extensionNames.size());

		vk::ApplicationInfo applicationInfo(AppName.c_str(), 1, EngineName.c_str(), 1, VK_API_VERSION_1_3);
		vk::InstanceCreateInfo instanceCreateInfo(
			vk::InstanceCreateFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR), 
			&applicationInfo,
			0,
			nullptr,
			extensionCount,
			extensionNames.data());

		instance = vk::createInstanceUnique(instanceCreateInfo);
	}

	VkSurfaceKHR vkSurface = VK_NULL_HANDLE;
	check_sdl(SDL_Vulkan_CreateSurface(window, instance.get(), &vkSurface));

	// Select a physical device
	vk::PhysicalDevice physicalDevice;
	{
		auto devices = instance->enumeratePhysicalDevices();
		std::cout << "Found " << devices.size() << " devices." << std::endl;
        
        
		const bool hasDiscreteGpu = std::find_if(devices.begin(), devices.end(),
		[](auto&& device)
		{
			auto properties = device.getProperties();
			return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
		}) != devices.end();
        
        std::cout << "Enumerating devices: " << std::endl;
        for (auto&& device : devices)
        {
            auto properties = device.getProperties();
            auto features = device.getFeatures();
            
            std::cout << properties.deviceName << std::endl;
            
            if (hasDiscreteGpu && properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
            {
                physicalDevice = device;
                break;
            }
            else if (!hasDiscreteGpu && properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            {
                physicalDevice = device;
                break;
            }
        }
        
        if (!physicalDevice)
        {
            throw std::runtime_error("No suitable physical device found!");
        }
	}

	// Create the device and queue
	vk::UniqueDevice device;
	vk::Queue queue;
    uint32_t graphicsQueueIndex = -1;
	{
        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
        {
            auto&& properties = queueFamilyProperties[i];
            auto presentSupport = physicalDevice.getSurfaceSupportKHR(i, vkSurface);
            if (presentSupport && properties.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                graphicsQueueIndex = i;
                break;
            }
        }
        std::cout << "Graphics queue index is " << graphicsQueueIndex << std::endl;
        const float queuePriority = 1.0f;

        vk::DeviceQueueCreateInfo queueCreateInfo(
			{},
			graphicsQueueIndex,
			1,
			&queuePriority);

		vk::PhysicalDeviceFeatures deviceFeatures;

		const std::vector<const char*> deviceExtensions = 
		{
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		};

		vk::DeviceCreateInfo createInfo(
			{}, 
			1, 
			&queueCreateInfo, 
			{}, 
			{}, 
			static_cast<uint32_t>(deviceExtensions.size()), 
			deviceExtensions.data(), 
			&deviceFeatures);
		device = physicalDevice.createDeviceUnique(createInfo);
        queue = device->getQueue(graphicsQueueIndex, 0);
	}
    
    // Create our swapchain
    vk::UniqueSwapchainKHR swapchain;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::UniqueImageView> swapchainImageViews;
    vk::Extent2D swapchainExtent(win_width, win_height);
    const vk::Format format = vk::Format::eB8G8R8A8Unorm;
    {
        vk::SwapchainCreateInfoKHR swapchainCreateInfo(
			{},
			vkSurface,
			2,
			format,
			vk::ColorSpaceKHR::eSrgbNonlinear,
			swapchainExtent,
			1,
			vk::ImageUsageFlagBits::eColorAttachment,
            vk::SharingMode::eExclusive,
			{},
			vk::SurfaceTransformFlagBitsKHR::eIdentity,
			vk::CompositeAlphaFlagBitsKHR::eOpaque,
			vk::PresentModeKHR::eFifo,
			true);
        swapchain = device->createSwapchainKHRUnique(swapchainCreateInfo);
        
        // Get the swapchain images
        swapchainImages = device->getSwapchainImagesKHR(swapchain.get());
        swapchainImageViews.reserve(swapchainImages.size());
        for (auto&& image : swapchainImages)
        {
            vk::ImageViewCreateInfo imageViewCreateInfo(
				{},
				image,
				vk::ImageViewType::e2D,
				format,
				{},
				vk::ImageSubresourceRange(
					vk::ImageAspectFlagBits::eColor,
					0,
					1,
					0,
					1));
            swapchainImageViews.push_back(std::move(device->createImageViewUnique(imageViewCreateInfo)));
        }
    }
    
    // Build the pipeline
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline pipeline;
	vk::UniquePipelineCache pipelineCache;
    vk::UniqueRenderPass renderPass;
    {
        // Vertex shader
        vk::ShaderModuleCreateInfo vertexShaderCreateInfo(
			{},
			sizeof(vert_spv),
			vert_spv);
        vk::UniqueShaderModule vertexShaderModule = device->createShaderModuleUnique(vertexShaderCreateInfo);
        
        // Fragment shader
        vk::ShaderModuleCreateInfo fragmentShaderCreateInfo(
            {},
            sizeof(frag_spv),
            frag_spv);
        vk::UniqueShaderModule fragmentShaderModule = device->createShaderModuleUnique(fragmentShaderCreateInfo);
        
        // Vertex stage
        vk::PipelineShaderStageCreateInfo vertexStage(
			{},
			vk::ShaderStageFlagBits::eVertex,
			vertexShaderModule.get(),
			"main");

        // Fragment stage
        vk::PipelineShaderStageCreateInfo fragmentStage(
			{},
			vk::ShaderStageFlagBits::eFragment,
			fragmentShaderModule.get(),
			"main");

		const std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = 
		{
			vertexStage,
			fragmentStage
		};

		// TODO: Don't hardcode vertices
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

		// Primitive type
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
			{},
			vk::PrimitiveTopology::eTriangleList,
			false);
        
		// Viewport
		vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(win_width), static_cast<float>(win_height), 0.0f, 1.0f);

		// Scissor rect
		vk::Rect2D scissor(vk::Offset2D(0.0f, 0.0f), swapchainExtent);

        vk::PipelineViewportStateCreateInfo viewportStateInfo(
			{},
			1,
			&viewport,
			1,
			&scissor);

		vk::PipelineRasterizationStateCreateInfo rasterizerInfo(
			{},
			false,
			false,
			vk::PolygonMode::eFill,
			vk::CullModeFlagBits::eBack,
			vk::FrontFace::eClockwise,
			false);

		vk::PipelineMultisampleStateCreateInfo multisampling(
			{},
			vk::SampleCountFlagBits::e1,
			false);

		vk::PipelineColorBlendAttachmentState blendMode(
			false,
			vk::BlendFactor::eZero,
			vk::BlendFactor::eZero,
			vk::BlendOp::eAdd,
			vk::BlendFactor::eZero,
			vk::BlendFactor::eZero,
			vk::BlendOp::eAdd,
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
		
		vk::PipelineColorBlendStateCreateInfo blendInfo(
			{},
			false,
			vk::LogicOp::eClear,
			1,
			&blendMode);

		vk::PipelineLayoutCreateInfo pipelineInfo;
		pipelineLayout = device->createPipelineLayoutUnique(pipelineInfo);
        
        vk::AttachmentDescription colorAttachment(
			{},
			format,
			vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::ePresentSrcKHR);
        
        vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);
        
		vk::SubpassDescription subpass(
			{},
			vk::PipelineBindPoint::eGraphics,
			{},
			{},
			1,
			&colorAttachmentRef);

        vk::RenderPassCreateInfo renderPassInfo(
			{},
			1,
			&colorAttachment,
			1,
			&subpass);
        renderPass = device->createRenderPassUnique(renderPassInfo);
        
		pipelineCache = device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo());

        vk::GraphicsPipelineCreateInfo graphicsPipelineInfo(
			{},
			2,
			shaderStages.data(),
			&vertexInputInfo,
			&inputAssembly,
			nullptr,
			&viewportStateInfo,
			&rasterizerInfo,
			&multisampling,
			nullptr,
			&blendInfo,
			nullptr,
			pipelineLayout.get(),
			renderPass.get(),
			0);
        pipeline = device->createGraphicsPipelineUnique(pipelineCache.get(), graphicsPipelineInfo).value;
    }
    
    // Setup framebuffers
    std::vector<vk::UniqueFramebuffer> framebuffers;
    framebuffers.reserve(swapchainImageViews.size());
    for (auto&& view : swapchainImageViews)
    {
		const std::vector<vk::ImageView> attachments = { view.get() };
        vk::FramebufferCreateInfo createInfo(
			{},
			renderPass.get(),
			1,
			attachments.data(),
			win_width,
			win_height,
			1);
        framebuffers.push_back(std::move(device->createFramebufferUnique(createInfo)));
    }

	// Setup the command pool
	VkCommandPool vk_command_pool;
	{
		VkCommandPoolCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		create_info.queueFamilyIndex = graphicsQueueIndex;
		check_vulkan(vkCreateCommandPool(device.get(), &create_info, nullptr, &vk_command_pool));
	}

	std::vector<VkCommandBuffer> command_buffers(framebuffers.size(), VkCommandBuffer{});
	{
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = vk_command_pool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = command_buffers.size();
		check_vulkan(vkAllocateCommandBuffers(device.get(), &info, command_buffers.data()));
	}

	// Now record the rendering commands (TODO: Could also do this pre-recording in the DXR backend
	// of rtobj. Will there be much perf. difference?)
	for (size_t i = 0; i < command_buffers.size(); ++i) 
	{
		auto& cmd_buf = command_buffers[i];

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		check_vulkan(vkBeginCommandBuffer(cmd_buf, &begin_info));

		VkRenderPassBeginInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_info.renderPass = renderPass.get();
		render_pass_info.framebuffer = framebuffers[i].get();
		render_pass_info.renderArea.offset.x = 0;
		render_pass_info.renderArea.offset.y = 0;
		render_pass_info.renderArea.extent = swapchainExtent;
		
		VkClearValue clear_color = { 0.f, 0.f, 0.f, 1.f };
		render_pass_info.clearValueCount = 1;
		render_pass_info.pClearValues = &clear_color;

		vkCmdBeginRenderPass(cmd_buf, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.get());

		// Draw our "triangle" embedded in the shader
		vkCmdDraw(cmd_buf, 3, 1, 0, 0);

		vkCmdEndRenderPass(cmd_buf);

		check_vulkan(vkEndCommandBuffer(cmd_buf));
	}

	VkSemaphore img_avail_semaphore = VK_NULL_HANDLE;
	VkSemaphore render_finished_semaphore = VK_NULL_HANDLE;
	{
		VkSemaphoreCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		check_vulkan(vkCreateSemaphore(device.get(), &info, nullptr, &img_avail_semaphore));
		check_vulkan(vkCreateSemaphore(device.get(), &info, nullptr, &render_finished_semaphore));
	}

	// We use a fence to wait for the rendering work to finish
	VkFence vk_fence = VK_NULL_HANDLE;
	{
		VkFenceCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		check_vulkan(vkCreateFence(device.get(), &info, nullptr, &vk_fence));
	}

	std::cout << "Running loop\n";
	bool done = false;
	while (!done) 
	{
		SDL_Event event;
		while (SDL_PollEvent(&event))
	    {
			if (event.type == SDL_QUIT) 
			{
				done = true;
			}
			if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) 
			{
				done = true;
			}
			if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE
					&& event.window.windowID == SDL_GetWindowID(window)) 
			{
				done = true;
			}
		}

		// Get an image from the swap chain
		uint32_t img_index = 0;
		check_vulkan(vkAcquireNextImageKHR(device.get(), swapchain.get(), std::numeric_limits<uint64_t>::max(),
			img_avail_semaphore, VK_NULL_HANDLE, &img_index));

		// We need to wait for the image before we can run the commands to draw to it, and signal
		// the render finished one when we're done
		const std::array<VkSemaphore, 1> wait_semaphores = { img_avail_semaphore };
		const std::array<VkSemaphore, 1> signal_semaphores = { render_finished_semaphore };
		const std::array<VkPipelineStageFlags, 1> wait_stages = { VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };

		check_vulkan(vkResetFences(device.get(), 1, &vk_fence));
		
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount = wait_semaphores.size();
		submit_info.pWaitSemaphores = wait_semaphores.data();
		submit_info.pWaitDstStageMask = wait_stages.data();
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers[img_index];
		submit_info.signalSemaphoreCount = signal_semaphores.size();
		submit_info.pSignalSemaphores = signal_semaphores.data();
		check_vulkan(vkQueueSubmit(queue, 1, &submit_info, vk_fence));

		// Finally, present the updated image in the swap chain
		std::array<VkSwapchainKHR, 1> present_chain = { swapchain.get() };
		VkPresentInfoKHR present_info = {};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = signal_semaphores.size();
		present_info.pWaitSemaphores = signal_semaphores.data();
		present_info.swapchainCount = present_chain.size();
		present_info.pSwapchains = present_chain.data();
		present_info.pImageIndices = &img_index;
		check_vulkan(vkQueuePresentKHR(queue, &present_info));

		// Wait for the frame to finish
		check_vulkan(vkWaitForFences(device.get(), 1, &vk_fence, true, std::numeric_limits<uint64_t>::max()));
	}

	vkDestroySemaphore(device.get(), img_avail_semaphore, nullptr);
	vkDestroySemaphore(device.get(), render_finished_semaphore, nullptr);
	vkDestroyFence(device.get(), vk_fence, nullptr);
	vkDestroyCommandPool(device.get(), vk_command_pool, nullptr);
	vkDestroySurfaceKHR(instance.get(), vkSurface, nullptr);

	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}

