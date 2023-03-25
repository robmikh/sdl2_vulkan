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
	VkPipelineLayout vk_pipeline_layout;
	VkRenderPass vk_render_pass;
	VkPipeline vk_pipeline;
	{
		VkShaderModule vertex_shader_module = VK_NULL_HANDLE;

		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = sizeof(vert_spv);
		create_info.pCode = vert_spv;
		check_vulkan(vkCreateShaderModule(device.get(), &create_info, nullptr, &vertex_shader_module));
		
		VkPipelineShaderStageCreateInfo vertex_stage = {};
		vertex_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertex_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertex_stage.module = vertex_shader_module;
		vertex_stage.pName = "main";

		VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = sizeof(frag_spv);
		create_info.pCode = frag_spv;
		check_vulkan(vkCreateShaderModule(device.get(), &create_info, nullptr, &fragment_shader_module));

		VkPipelineShaderStageCreateInfo fragment_stage = {};
		fragment_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragment_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragment_stage.module = fragment_shader_module;
		fragment_stage.pName = "main";

		std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = { vertex_stage, fragment_stage };

		// Vertex data hard-coded in vertex shader
		VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.vertexBindingDescriptionCount = 0;
		vertex_input_info.vertexAttributeDescriptionCount = 0;

		// Primitive type
		VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly.primitiveRestartEnable = VK_FALSE;

		// Viewport config
		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = win_width;
		viewport.height = win_height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		// Scissor rect config
		VkRect2D scissor = {};
		scissor.offset.x = 0;
		scissor.offset.y = 0;
		scissor.extent = swapchainExtent;

		VkPipelineViewportStateCreateInfo viewport_state_info = {};
		viewport_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state_info.viewportCount = 1;
		viewport_state_info.pViewports = &viewport;
		viewport_state_info.scissorCount = 1;
		viewport_state_info.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer_info = {};
		rasterizer_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer_info.depthClampEnable = VK_FALSE;
		rasterizer_info.rasterizerDiscardEnable = VK_FALSE;
		rasterizer_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer_info.lineWidth = 1.f;
		rasterizer_info.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer_info.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blend_mode = {};
		blend_mode.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		blend_mode.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo blend_info = {};
		blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blend_info.logicOpEnable = VK_FALSE;
		blend_info.attachmentCount = 1;
		blend_info.pAttachments = &blend_mode;

		VkPipelineLayoutCreateInfo pipeline_info = {};
		pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		check_vulkan(vkCreatePipelineLayout(device.get(), &pipeline_info, nullptr, &vk_pipeline_layout));

		VkAttachmentDescription color_attachment = {};
		color_attachment.format = static_cast<VkFormat>(format);
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

		VkRenderPassCreateInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 1;
		render_pass_info.pAttachments = &color_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;
		check_vulkan(vkCreateRenderPass(device.get(), &render_pass_info, nullptr, &vk_render_pass));

		VkGraphicsPipelineCreateInfo graphics_pipeline_info = {};
		graphics_pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		graphics_pipeline_info.stageCount = 2;
		graphics_pipeline_info.pStages = shader_stages.data();
		graphics_pipeline_info.pVertexInputState = &vertex_input_info;
		graphics_pipeline_info.pInputAssemblyState = &input_assembly;
		graphics_pipeline_info.pViewportState = &viewport_state_info;
		graphics_pipeline_info.pRasterizationState = &rasterizer_info;
		graphics_pipeline_info.pMultisampleState = &multisampling;
		graphics_pipeline_info.pColorBlendState = &blend_info;
		graphics_pipeline_info.layout = vk_pipeline_layout;
		graphics_pipeline_info.renderPass = vk_render_pass;
		graphics_pipeline_info.subpass = 0;
		check_vulkan(vkCreateGraphicsPipelines(device.get(), VK_NULL_HANDLE, 1, &graphics_pipeline_info, nullptr, &vk_pipeline));

		vkDestroyShaderModule(device.get(), vertex_shader_module, nullptr);
		vkDestroyShaderModule(device.get(), fragment_shader_module, nullptr);
	}

	// Setup framebuffers
	std::vector<VkFramebuffer> framebuffers;
	for (const auto &v : swapchainImageViews)
	{
		std::array<VkImageView, 1> attachments = { v.get() };
		VkFramebufferCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		create_info.renderPass = vk_render_pass;
		create_info.attachmentCount = 1;
		create_info.pAttachments = attachments.data();
		create_info.width = win_width;
		create_info.height = win_height;
		create_info.layers = 1;
		VkFramebuffer fb = VK_NULL_HANDLE;
		check_vulkan(vkCreateFramebuffer(device.get(), &create_info, nullptr, &fb));
		framebuffers.push_back(fb);
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
		render_pass_info.renderPass = vk_render_pass;
		render_pass_info.framebuffer = framebuffers[i];
		render_pass_info.renderArea.offset.x = 0;
		render_pass_info.renderArea.offset.y = 0;
		render_pass_info.renderArea.extent = swapchainExtent;
		
		VkClearValue clear_color = { 0.f, 0.f, 0.f, 1.f };
		render_pass_info.clearValueCount = 1;
		render_pass_info.pClearValues = &clear_color;

		vkCmdBeginRenderPass(cmd_buf, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

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
	for (auto &fb : framebuffers) 
	{
		vkDestroyFramebuffer(device.get(), fb, nullptr);
	}
	vkDestroyPipeline(device.get(), vk_pipeline, nullptr);
	vkDestroyRenderPass(device.get(), vk_render_pass, nullptr);
	vkDestroyPipelineLayout(device.get(), vk_pipeline_layout, nullptr);
	vkDestroySurfaceKHR(instance.get(), vkSurface, nullptr);

	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}

