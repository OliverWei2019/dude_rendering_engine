
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_types.h>
#include <vk_initializers.h>
#include <vk_descriptors.h>

#include "VkBootstrap.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <sstream>
#include "vk_textures.h"
#include "vk_shaders.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"
#include "prefab_asset.h"
#include "material_asset.h"

#include "Tracy.hpp"
#include "TracyVulkan.hpp"
#include "vk_profiler.h"

#include "fmt/core.h"
#include "fmt/os.h"
#include "fmt/color.h"

#include "logger.h"
#include "cvars.h"



AutoCVar_Int CVAR_OcclusionCullGPU("culling.enableOcclusionGPU", "Perform occlusion culling in gpu", 1, CVarFlags::EditCheckbox);


AutoCVar_Int CVAR_CamLock("camera.lock", "Locks the camera", true, CVarFlags::EditCheckbox);
AutoCVar_Int CVAR_OutputIndirectToFile("culling.outputIndirectBufferToFile", "output the indirect data to a file. Autoresets", 0, CVarFlags::EditCheckbox);

AutoCVar_Float CVAR_DrawDistance("gpu.drawDistance", "Distance cull", 5000);

AutoCVar_Int CVAR_FreezeShadows("gpu.freezeShadows", "Stop the rendering of shadows", 0, CVarFlags::EditCheckbox);


constexpr bool bUseValidationLayers = true;

//we want to immediately abort when there is an error. In normal engines this would give an error message to the user, or perform a dump of state.
using namespace std;
#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = x;                                           \
		if (err)                                                    \
		{                                                           \
			std::cout <<"Detected Vulkan error: " << err << std::endl; \
			abort();                                                \
		}                                                           \
	} while (0)




void VulkanEngine::init()
{
	ZoneScopedN("Engine Init");	
	//setup logger and set start time point;
	LogHandler::Get().set_time();

	LOG_INFO("Engine Init");

	// We initialize SDL and create a window with it. 
	SDL_Init(SDL_INIT_VIDEO);
	LOG_SUCCESS("SDL inited");
	SDL_WindowFlags window_flags = (SDL_WINDOW_VULKAN);

	_window = SDL_CreateWindow(
		"Engine Demo!",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_windowExtent.width,
		_windowExtent.height,
		window_flags
	);

	//_renderables.reserve(10000);
	//reserve 1000 meshs
	_meshes.reserve(1000);
	
	init_vulkan();
	//open the pipeline profiler and grab pipeline state info
	_profiler = new vkutil::VulkanProfiler();

	_profiler->init(_device, _gpuProperties.limits.timestampPeriod);
	_mainDeletionQueue.push_function([&]() {
		_profiler->cleanup();
		delete _profiler;
		});
	//initailizes shader cache (device)
	_shaderCache.init(_device);
	_mainDeletionQueue.push_function([=]() {
		_shaderCache.cleanup();
		
		});
	//register pass type based mesh types
	_renderScene.init();
	_mainDeletionQueue.push_function([&]() {
		_renderScene.flush(this);
		});
	init_swapchain();


	init_forward_renderpass();
	init_copy_renderpass();
	init_shadow_renderpass();

	init_framebuffers();

	init_commands();

	init_sync_structures();

	init_descriptors();

	init_pipelines();

	LOG_INFO("Engine Initialized, starting Load");
	

	load_images();

	//load_meshes();

	init_scene();

	LOG_INFO("Scene and Material initializated");

	init_imgui();
	
	adjust_image_layout();

	_renderScene.build_batches();

	_renderScene.merge_meshes(this);
	//everything went fine
	_isInitialized = true;

	_camera = {};
	//Sponza 
	_camera.position = { 800.f,800.f,0.f };

	_mainLight.lightPosition = { 800,1500,400 };
	_mainLight.lightDirection = glm::vec3(1, -10, -10);
	_mainLight.shadowExtent = { 500 ,500 ,500 };
	

	////FlightHelmet
	//_camera.position = { 0.f,5.f,3.f };

	//_mainLight.lightPosition = { 10,100,40 };
	//_mainLight.lightDirection = glm::vec3(1, -10, -10);
	//_mainLight.shadowExtent = { 100 ,100 ,100 };
}

void VulkanEngine::cleanup()
{
	if (_isInitialized) {

		//make sure the gpu has stopped doing its things
		LOG_INFO("Cleanup resource");
		vkDeviceWaitIdle(_device);
		for (auto& frame : _frames)
		{
			vkWaitForFences(_device, 1, &frame._renderFence, true, 1000000000);
		}
		for (auto& frame : _frames)
		{
			//frame.
			frame.dynamicDescriptorAllocator->cleanup();
			vmaDestroyBuffer(_allocator, frame.debugOutputBuffer._buffer, frame.debugOutputBuffer._allocation);
			frame.dynamicData.cleanup(_allocator, frame.dynamicData.source);
			//vmaDestroyBuffer(_allocator, frame.dynamicData.source._buffer, frame.dynamicData.source._allocation);

			frame._frameDeletionQueue.flush();
		}

		_mainDeletionQueue.flush();
		//destroy loaded texture image views
		/*for (auto Text : _loadedTextures) {
			auto text = Text.second;
			if (text.image._image && text.image._defaultView) {
				vmaDestroyImage(_allocator, text.image._image, text.image._allocation);
				vkDestroyImageView(_device, text.image._defaultView, nullptr);
			}
			if (text.imageView) {
				vkDestroyImageView(_device, text.imageView, nullptr);
			}
		}*/
		for (auto Mesh : _meshes) {
			auto mesh = Mesh.second;
			if (mesh._vertexBuffer._buffer) {
				vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
			}
			if (mesh._indexBuffer._buffer) {
				vmaDestroyBuffer(_allocator, mesh._indexBuffer._buffer, mesh._indexBuffer._allocation);
			}
		}

		_descriptorAllocator->cleanup();
		_descriptorLayoutCache->cleanup();

		vmaDestroyAllocator(_allocator);
		vkDestroyDevice(_device, nullptr);

		if (_debug_messenger) {
			vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
		}
		vkDestroySurfaceKHR(_instance, _surface, nullptr);

		vkDestroyInstance(_instance, nullptr);

		SDL_DestroyWindow(_window);
		LOG_SUCCESS("Empty all resource");
	}
}


void VulkanEngine::draw()
{
	ZoneScopedN("Engine Draw");

	ImGui::Render();

	{
		ZoneScopedN("Fence Wait");
		//wait until the gpu has finished rendering the last frame. Timeout of 1 second
		VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, 1000000000));
		VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

		//reflesh 3 render pass(forward pass,shadow pass,transparency pass)
		_renderScene.build_batches();
	
		//reset push buffer (dynamic data) wait re-fill data
		get_current_frame().dynamicData.reset();
		//check the debug data
		void* data;		
		vmaMapMemory(_allocator, get_current_frame().debugOutputBuffer._allocation, &data);
		for (int i =1 ; i <   get_current_frame().debugDataNames.size();i++)
		{
			//debug data storage range
			uint32_t begin = get_current_frame().debugDataOffsets[i-1];
			uint32_t end = get_current_frame().debugDataOffsets[i];
			
			//debugDataName -> record bugges type
			auto name = get_current_frame().debugDataNames[i];
			if (name.compare("Cull Indirect Output") == 0)
			{
				//malloc copy memeory
				void* buffer = malloc(end - begin);
				//copy debug buffer data to host buffer(not vulkan buffer)
				memcpy(buffer, (uint8_t*)data + begin, end - begin);
				//Split memory by GPUIndirectObject object size
				GPUIndirectObject* objects = (GPUIndirectObject*)buffer;
				int objectCount = (end - begin) / sizeof(GPUIndirectObject);

				//record debug info: _frameNumber->frames index,i->GPUindirectObject array first index
				std::string filename = fmt::format("{}_CULLDATA_{}.txt", _frameNumber,i);
				//storage debug info to file and output
				auto out = fmt::output_file(filename);

				//GPUindirectObject count->renderable batches index
				//GPUindirectObject <- renderable batches convert
				for (int o = 0; o < objectCount; o++)
				{
					out.print("DRAW: {} ------------ \n", o);
					//renderable object in one batch(GPUindirectObject) count
					out.print("	OG Count: {} \n", _renderScene._forwardPass.batches[o].count);
					//instancing draw object count
					out.print("	Visible Count: {} \n", objects[o].command.instanceCount);
					out.print("	First: {} \n", objects[o].command.firstInstance);
					out.print("	Indices: {} \n", objects[o].command.indexCount);
					
				}	
				
				free(buffer);
			}
		}

		vmaUnmapMemory(_allocator, get_current_frame().debugOutputBuffer._allocation);
		
		get_current_frame().debugDataNames.clear();
		get_current_frame().debugDataOffsets.clear();

		get_current_frame().debugDataNames.push_back("");
		get_current_frame().debugDataOffsets.push_back(0);

		//clear previous frame souce flush memory
		get_current_frame()._frameDeletionQueue.flush();
		get_current_frame().dynamicDescriptorAllocator->reset_pools();
	}

	//clear previous frame souce flush memory
	//get_current_frame()._frameDeletionQueue.flush();
	//get_current_frame().dynamicDescriptorAllocator->reset_pools();

	//now that we are sure that the commands finished executing, 
	//we can safely reset the command buffer to begin recording again.
	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));
	uint32_t swapchainImageIndex;
	{
		ZoneScopedN("Aquire Image");
		//request image and index from the swapchain
	
		VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 0, get_current_frame()._presentSemaphore, nullptr, &swapchainImageIndex));

	}

	//naming it cmd for shorter writing
	VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	//make a clear-color from frame number. This will flash with a 120 frame period.
	VkClearValue clearValue;
	float flash = abs(sin(_frameNumber / 120.f));
	clearValue.color = { { 0.1f, 0.1f, 0.1f, 1.0f } };
	
	//get timestamp and state from query pool
	_profiler->grab_queries(cmd);

	{

		postCullBarriers.clear();
		cullReadyBarriers.clear();

		TracyVkZone(_graphicsQueueContext, get_current_frame()._mainCommandBuffer, "All Frame");
		ZoneScopedNC("Render Frame", tracy::Color::White);

		vkutil::VulkanScopeTimer timer(cmd, _profiler, "All Frame");

		{
			vkutil::VulkanScopeTimer timer2(cmd, _profiler, "Ready Frame");
			// ready for scene objects and pass copy to GPU using upload pipeline
			// reflesh indirect object source array and update pass indirect indices array 
			ready_mesh_draw(cmd);

			
			//copy clear dirty objects array to draw indirect objects array
			//set and binding barriers
			ready_cull_data(_renderScene._forwardPass, cmd);
			ready_cull_data(_renderScene._transparentForwardPass, cmd);
			ready_cull_data(_renderScene._shadowPass, cmd);

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, cullReadyBarriers.size(), cullReadyBarriers.data(), 0, nullptr);
		}

		//Before execute pass rendering, execute cull compute pipeline, cull unuseful data
		CullParams forwardCull;
		forwardCull.projmat = _camera.get_projection_matrix(true);
		forwardCull.viewmat = _camera.get_view_matrix();
		forwardCull.frustrumCull = true;
		forwardCull.occlusionCull = true;
		forwardCull.drawDist = CVAR_DrawDistance.Get();
		forwardCull.aabb = false;
		{
			execute_compute_cull(cmd, _renderScene._forwardPass, forwardCull);
			execute_compute_cull(cmd, _renderScene._transparentForwardPass, forwardCull);
		}

		glm::vec3 extent = _mainLight.shadowExtent * 10.f;
		glm::mat4 projection = glm::orthoLH_ZO(-extent.x, extent.x, -extent.y, extent.y, -extent.z, extent.z);
		
		
		CullParams shadowCull;
		shadowCull.projmat = _mainLight.get_projection();
		shadowCull.viewmat = _mainLight.get_view();
		shadowCull.frustrumCull = true;
		shadowCull.occlusionCull = false;
		shadowCull.drawDist = 9999999;
		shadowCull.aabb = true;

		glm::vec3 aabbcenter = _mainLight.lightPosition;
		glm::vec3 aabbextent = _mainLight.shadowExtent * 1.5f;
		shadowCull.aabbmax = aabbcenter + aabbextent;
		shadowCull.aabbmin = aabbcenter - aabbextent;

		{
			vkutil::VulkanScopeTimer timer2(cmd, _profiler, "Shadow Cull");

			if (*CVarSystem::Get()->GetIntCVar("gpu.shadowcast"))
			{
				execute_compute_cull(cmd, _renderScene._shadowPass, shadowCull);
			}
		}

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, postCullBarriers.size(), postCullBarriers.data(), 0, nullptr);


		//execute pass rendering
		shadow_pass(cmd);
		
		forward_pass(clearValue, cmd);
		
		reduce_depth(cmd);

		copy_render_to_swapchain(swapchainImageIndex, cmd);
	}

	TracyVkCollect(_graphicsQueueContext, get_current_frame()._mainCommandBuffer);

	//finalize the command buffer (we can no longer add commands, but it can now be executed)
	VK_CHECK(vkEndCommandBuffer(cmd));

	//prepare the submission to the queue. 
	//we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
	//we will signal the _renderSemaphore, to signal that rendering has finished

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &get_current_frame()._presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &get_current_frame()._renderSemaphore;
	{
		ZoneScopedN("Queue Submit");
		//submit command buffer to the queue and execute it.
		// _renderFence will now block until the graphic commands finish execution
		VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

	}
	//prepare present
	// this will put the image we just rendered to into the visible window.
	// we want to wait on the _renderSemaphore for that, 
	// as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo = vkinit::present_info();

	presentInfo.pSwapchains = &_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	{
		ZoneScopedN("Queue Present");
		VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

	}
	//increase the number of frames drawn
	_frameNumber++;
}


void VulkanEngine::forward_pass(VkClearValue clearValue, VkCommandBuffer cmd)
{
	vkutil::VulkanScopeTimer timer(cmd, _profiler, "Forward Pass");
	vkutil::VulkanPipelineStatRecorder timer2(cmd, _profiler, "Forward Primitives");
	//clear depth at 0
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 0.f;

	//start the main renderpass. 
	//We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	//! forwardFrameBuffer include two imageview !
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_renderPass, _windowExtent, _forwardFramebuffer/*_framebuffers[swapchainImageIndex]*/);

	//connect clear values
	rpInfo.clearValueCount = 2;
	//clear value->clear color,depth clear->clear depth value
	VkClearValue clearValues[] = { clearValue, depthClear };

	rpInfo.pClearValues = &clearValues[0];
	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)_windowExtent.width;
	viewport.height = (float)_windowExtent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = _windowExtent;

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	vkCmdSetDepthBias(cmd, 0, 0, 0);


	//stats.drawcalls = 0;
	//stats.draws = 0;
	//stats.objects = 0;
	//stats.triangles = 0;

	{
		TracyVkZone(_graphicsQueueContext, get_current_frame()._mainCommandBuffer, "Forward Pass");
		//draw_objects_forward(cmd, _renderScene._transparentForwardPass);
		draw_objects_forward(cmd, _renderScene._forwardPass);
		draw_objects_forward(cmd, _renderScene._transparentForwardPass);
	}

	//imgui draw meau UI
	{
		TracyVkZone(_graphicsQueueContext, get_current_frame()._mainCommandBuffer, "Imgui Draw");
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
	}

	//finalize the render pass
	vkCmdEndRenderPass(cmd);
}


void VulkanEngine::shadow_pass(VkCommandBuffer cmd)
{
	

	vkutil::VulkanScopeTimer timer(cmd, _profiler, "Shadow Pass");
	vkutil::VulkanPipelineStatRecorder timer2(cmd, _profiler, "Shadow Primitives");
	if (CVAR_FreezeShadows.Get()) return;
	if (!*CVarSystem::Get()->GetIntCVar("gpu.shadowcast"))
	{
		return;
	}

	//clear depth at 1
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.f;	
	VkRenderPassBeginInfo rpInfo = vkinit::renderpass_begin_info(_shadowPass, _shadowExtent, _shadowFramebuffer);

	//connect clear values
	rpInfo.clearValueCount = 1;

	VkClearValue clearValues[] = { depthClear };

	rpInfo.pClearValues = &clearValues[0];
	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)_shadowExtent.width;
	viewport.height = (float)_shadowExtent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = _shadowExtent;

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	


	stats.drawcalls = 0;
	stats.draws = 0;
	stats.objects = 0;
	stats.triangles = 0;

	if(_renderScene._shadowPass.batches.size() > 0)
	{
		TracyVkZone(_graphicsQueueContext, get_current_frame()._mainCommandBuffer, "Shadow  Pass");
		draw_objects_shadow(cmd, _renderScene._shadowPass);
	}

	//finalize the render pass
	vkCmdEndRenderPass(cmd);
}

void VulkanEngine::copy_render_to_swapchain(uint32_t swapchainImageIndex, VkCommandBuffer cmd)
{
	//start the main renderpass. 
	//We will use the clear color from above, and the framebuffer of the index the swapchain gave us
	//initalize copy pass
	VkRenderPassBeginInfo copyRP = vkinit::renderpass_begin_info(_copyPass, _windowExtent, _framebuffers[swapchainImageIndex]);


	vkCmdBeginRenderPass(cmd, &copyRP, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)_windowExtent.width;
	viewport.height = (float)_windowExtent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = _windowExtent;

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdSetDepthBias(cmd, 0, 0, 0);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blitPipeline);

	VkDescriptorImageInfo sourceImage;
	sourceImage.sampler = _smoothSampler;

	sourceImage.imageView = _rawRenderImage._defaultView;
	sourceImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkDescriptorSet blitSet;
	vkutil::DescriptorBuilder::begin(_descriptorLayoutCache, get_current_frame().dynamicDescriptorAllocator)
		.bind_image(0, &sourceImage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.build(blitSet);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _blitLayout, 0, 1, &blitSet, 0, nullptr);

	vkCmdDraw(cmd, 3, 1, 0, 0);


	vkCmdEndRenderPass(cmd);
}

void VulkanEngine::run()
{

	LOG_INFO("Starting Main Loop ");
	
	bool bQuit = false;

	// Using time point and system_clock 
	std::chrono::time_point<std::chrono::system_clock> start, end;
	
	start = std::chrono::system_clock::now();
	end = std::chrono::system_clock::now();
	//main loop
	while (!bQuit)
	{
		ZoneScopedN("Main Loop");
		end = std::chrono::system_clock::now();
		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		//std::chrono::duration<float> elapsed_seconds = end - start;
		stats.frametime = elapsed_seconds.count();

		start = std::chrono::system_clock::now();
		//Handle events on queue
		SDL_Event e;
		{
			ZoneScopedNC("Event Loop", tracy::Color::White);
		while (SDL_PollEvent(&e) != 0)
		{

			ImGui_ImplSDL2_ProcessEvent(&e);
			//process key event change camera class input arg array
			_camera.process_input_event(&e);

			//close the window when user alt-f4s or clicks the X button			
			if (e.type == SDL_QUIT)
			{
				bQuit = true;
			}
			else if (e.type == SDL_KEYDOWN)
			{
				if (e.key.keysym.sym == SDLK_SPACE)
				{
					_selectedShader += 1;
					if (_selectedShader > 1)
					{
						_selectedShader = 0;
					}
				}
				if (e.key.keysym.sym == SDLK_TAB)
				{
					if (CVAR_CamLock.Get())
					{
						LOG_INFO("Mouselook disabled");
						CVAR_CamLock.Set(false);
					}
					else {
						LOG_INFO("Mouselook enabled");
						CVAR_CamLock.Set(true);
					}
				}
			}
		}
		}
		{

			{
				ZoneScopedNC("Imgui Logic", tracy::Color::Grey);

				//imgui new frame 
				ImGui_ImplVulkan_NewFrame();
				ImGui_ImplSDL2_NewFrame(_window);

				ImGui::NewFrame();



				if (ImGui::BeginMainMenuBar())
				{

					if (ImGui::BeginMenu("Engine"))
					{
						if (ImGui::BeginMenu("CVAR"))
						{
							CVarSystem::Get()->DrawImguiEditor();
							ImGui::EndMenu();
						}
						ImGui::EndMenu();
					}
					ImGui::EndMainMenuBar();
				}


				ImGui::Begin("Debug Menu");

				ImGui::Text("Frametimes: %f ms",stats.frametime);
				ImGui::Text("FPS: %f", 1.0f / (stats.frametime / 1000.0f));
				ImGui::Text("Objects: %d", stats.objects);
				ImGui::Text("Drawcalls: %d", stats.drawcalls);
				ImGui::Text("Batches: %d", stats.draws);
				ImGui::Text("Triangles: %d", stats.triangles);

				CVAR_OutputIndirectToFile.Set(false);
				if (ImGui::Button("Output Indirect"))
				{
					CVAR_OutputIndirectToFile.Set(true);
				}


				ImGui::Separator();

				for (auto& [k, v] : _profiler->timing)
				{
					ImGui::Text("TIME %s %f ms", k.c_str(), v);
				}
				for (auto& [k, v] : _profiler->stats)
				{
					ImGui::Text("STAT %s %d", k.c_str(), v);
				}


				ImGui::End();
			}

			{
				ZoneScopedNC("Flag Objects", tracy::Color::Blue);
				//test flagging some objects for changes

				int N_changes = 1000;
				for (int i = 0; i < N_changes; i++)
				{
					int rng = rand() % _renderScene.renderables.size();

					Handle<RenderObject> h;
					h.handle = rng;
					_renderScene.update_object(h);
				}
				_camera.bLocked = CVAR_CamLock.Get();
				//change camera position
				_camera.update_camera(stats.frametime);
				// direction light position
				_mainLight.lightPosition = _camera.position;
			}

			draw();
		}
	}
}

FrameData& VulkanEngine::get_current_frame()
{
	return _frames[_frameNumber % FRAME_OVERLAP];
}


FrameData& VulkanEngine::get_last_frame()
{
	return _frames[(_frameNumber - 1) % 2];
}


void VulkanEngine::process_input_event(SDL_Event* ev)
{
	if (ev->type == SDL_KEYDOWN)
	{
		switch (ev->key.keysym.sym)
		{		
		
		}
	}
	else if (ev->type == SDL_KEYUP)
	{
		switch (ev->key.keysym.sym)
		{
		case SDLK_UP:
		case SDLK_w:
			_camera.inputAxis.x -= 1.f;
			break;
		case SDLK_DOWN:
		case SDLK_s:
			_camera.inputAxis.x += 1.f;
			break;
		case SDLK_LEFT:
		case SDLK_a:
			_camera.inputAxis.y += 1.f;
			break;
		case SDLK_RIGHT:
		case SDLK_d:
			_camera.inputAxis.y -= 1.f;
			break;
		}
	}
	else if (ev->type == SDL_MOUSEMOTION) {
		if (!CVAR_CamLock.Get())
		{
			_camera.pitch -= ev->motion.yrel * 0.003f;
			_camera.yaw -= ev->motion.xrel * 0.003f;
		}
	}

	_camera.inputAxis = glm::clamp(_camera.inputAxis, { -1.0,-1.0,-1.0 }, { 1.0,1.0,1.0 });
}

void VulkanEngine::init_vulkan()
{
	
	vkb::InstanceBuilder builder;
	//make the vulkan instance, with basic debug features
	auto inst_ret = builder.set_app_name("Example Vulkan Application")
		.require_api_version(1, 2, 0)
		.request_validation_layers(bUseValidationLayers)
		.use_default_debug_messenger()
		.build();


	LOG_SUCCESS("Vulkan Instance initialized");

	vkb::Instance vkb_inst = inst_ret.value();

	//grab the instance 
	_instance = vkb_inst.instance;
	_debug_messenger = vkb_inst.debug_messenger;
	
	SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

	LOG_SUCCESS("SDL Surface initialized");

	//use vkbootstrap to select a gpu. 
	//We want a gpu that can write to the SDL surface and supports vulkan 1.2
	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	VkPhysicalDeviceFeatures feats{};

	feats.pipelineStatisticsQuery = true;
	feats.multiDrawIndirect = true;
	feats.drawIndirectFirstInstance = true;
	feats.samplerAnisotropy = true;
	selector.set_required_features(feats);

	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 2)
		.set_surface(_surface)
		//.add_required_extension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME)
		//.add_required_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)
		.select()
		.value();

	LOG_SUCCESS("GPU found");

	//create the final vulkan device

	vkb::DeviceBuilder deviceBuilder{ physicalDevice };


	

	vkb::Device vkbDevice = deviceBuilder.build().value();
	
	// Get the VkDevice handle used in the rest of a vulkan application
	_device = vkbDevice.device;
	_chosenGPU = physicalDevice.physical_device;

	// use vkbootstrap to get a Graphics queue
	_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	//initialize the memory allocator
	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = _chosenGPU;
	allocatorInfo.device = _device;
	allocatorInfo.instance = _instance;
	vmaCreateAllocator(&allocatorInfo, &_allocator);


	
	vkGetPhysicalDeviceProperties(_chosenGPU, &_gpuProperties);

	LOG_INFO("The gpu has a minimum buffer alignement of {}", _gpuProperties.limits.minUniformBufferOffsetAlignment);
	VkSampleCountFlags counts = _gpuProperties.limits.framebufferColorSampleCounts & _gpuProperties.limits.framebufferDepthSampleCounts;
	if (counts & VK_SAMPLE_COUNT_64_BIT) { msaaSampleCount = VK_SAMPLE_COUNT_64_BIT; }
	else if (counts & VK_SAMPLE_COUNT_32_BIT) { msaaSampleCount = VK_SAMPLE_COUNT_32_BIT; }
	else if (counts & VK_SAMPLE_COUNT_16_BIT) { msaaSampleCount = VK_SAMPLE_COUNT_16_BIT; }
	else if (counts & VK_SAMPLE_COUNT_8_BIT) { msaaSampleCount = VK_SAMPLE_COUNT_8_BIT; }
	else if (counts & VK_SAMPLE_COUNT_4_BIT) { msaaSampleCount = VK_SAMPLE_COUNT_4_BIT; }
	else if (counts & VK_SAMPLE_COUNT_2_BIT) { msaaSampleCount = VK_SAMPLE_COUNT_2_BIT; }
	else {
		msaaSampleCount = VK_SAMPLE_COUNT_1_BIT;
	}
	LOG_INFO("The engine setup msaa sample count {}", msaaSampleCount);
}
uint32_t previousPow2(uint32_t v)
{
	//Find the largest power of 2 less than a certain v
	uint32_t r = 1;

	while (r * 2 < v)
		r *= 2;
	//while ((r << 1) < v)
	//	r << 1;
	return r;
}
uint32_t getImageMipLevels(uint32_t width, uint32_t height)
{
	uint32_t result = 1;

	while (width > 1 || height > 1)
	{
		result++;
		width /= 2;
		height /= 2;
		
	}

	return result;
}
void VulkanEngine::init_swapchain()
{

	vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU,_device,_surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		//use vsync present mode
		.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
		.set_desired_extent(_windowExtent.width, _windowExtent.height)
		
		.build()
		.value();

	//store swapchain and its related images
	_swapchain = vkbSwapchain.swapchain;

	_mainDeletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
		});

	_swapchainImages = vkbSwapchain.get_images().value();
	_swapchainImageViews = vkbSwapchain.get_image_views().value();

	_swachainImageFormat = vkbSwapchain.image_format;
	//color resource image
	VkExtent3D renderImageExtent = {
			_windowExtent.width,
			_windowExtent.height,
			1
	};
	
	VkImageCreateInfo re_info = vkinit::image_create_info(_swachainImageFormat, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, renderImageExtent);
	re_info.samples = msaaSampleCount;
	VmaAllocationCreateInfo reimg_allocinfo = {};
	reimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	reimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	//allocate and create the color resource image
	VK_CHECK(vmaCreateImage(_allocator, &re_info, &reimg_allocinfo, &_colorResourceImage._image, &_colorResourceImage._allocation, nullptr));
	
	//build a color resource image-view to use for rendering
	VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_swachainImageFormat, _colorResourceImage._image, VK_IMAGE_ASPECT_COLOR_BIT);
	
	VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_colorResourceImage._defaultView));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _colorResourceImage._defaultView, nullptr);
		vmaDestroyImage(_allocator, _colorResourceImage._image, _colorResourceImage._allocation);
		});

	//render image
	{
		_renderFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
		VkImageCreateInfo ri_info = vkinit::image_create_info(_renderFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT| VK_IMAGE_USAGE_SAMPLED_BIT, renderImageExtent);
		//ri_info.samples = msaaSampleCount;

		//for the depth image, we want to allocate it from gpu local memory
		VmaAllocationCreateInfo dimg_allocinfo = {};
		dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		//allocate and create the undeal raw image
		VK_CHECK(vmaCreateImage(_allocator, &ri_info, &dimg_allocinfo, &_rawRenderImage._image, &_rawRenderImage._allocation, nullptr));

		//build a raw image-view for the depth image to use for rendering
		VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_renderFormat, _rawRenderImage._image, VK_IMAGE_ASPECT_COLOR_BIT);

		VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_rawRenderImage._defaultView));
		
		_mainDeletionQueue.push_function([=]() {
			
			vkDestroyImageView(_device, _rawRenderImage._defaultView, nullptr);
			vmaDestroyImage(_allocator, _rawRenderImage._image, _rawRenderImage._allocation);
			
			});
	}

	//depth image size will match the window
	VkExtent3D depthImageExtent = {
		_windowExtent.width,
		_windowExtent.height,
		1
	};
	//1024*4,1024*4
	VkExtent3D shadowExtent = {
		_shadowExtent.width,
		_shadowExtent.height,
		1
	};

	//hardcoding the depth format to 32 bit float
	_depthFormat = VK_FORMAT_D32_SFLOAT;

	//for the depth image, we want to allocate it from gpu local memory
	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	// depth image ------ 
	{
		//the depth image will be a image with the format we selected and Depth Attachment usage flag
		VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, depthImageExtent);
		//dimg_info.samples = msaaSampleCount;

		//allocate and create the image
		VK_CHECK(vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr));


		//build a image-view for the depth image to use for rendering
		VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);;

		VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage._defaultView));
		_mainDeletionQueue.push_function([=]() {
			vkDestroyImageView(_device, _depthImage._defaultView, nullptr);
			vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);

			});
	}
	//shadow image
	{
		//the depth image will be a image with the format we selected and Depth Attachment usage flag
		//width,height -> 1024*4,1024*4
		VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shadowExtent);
		//dimg_info.samples = msaaSampleCount;
		//allocate and create the image
		VK_CHECK(vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_shadowImage._image, &_shadowImage._allocation, nullptr));

		//build a image-view for the depth image to use for rendering
		VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _shadowImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

		VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_shadowImage._defaultView));
		_mainDeletionQueue.push_function([=]() {
			vkDestroyImageView(_device, _shadowImage._defaultView, nullptr);
			vmaDestroyImage(_allocator, _shadowImage._image, _shadowImage._allocation);
			
			});
	}

	//Hierarchical z-buffer Occlusion Culling
	/*
	* The normal depth buffer is too detailed for doing efficient culling, 
	* so we need to convert it into a depth pyramid. 
	* The idea of a depth pyramid is that we build a mipmap chain for the depth buffer in a way that the depth values are allways the maximum depth of that region. 
	* That way, we can look up directly into a mipmap so that pixel size is similar to object size, 
	* and it gives us a fairly accurate approximation.
	* 
	*/
	// Note: previousPow2 makes sure all reductions are at most by 2x2 which makes sure they are conservative
	depthPyramidWidth = previousPow2(_windowExtent.width);
	depthPyramidHeight = previousPow2(_windowExtent.height);
	depthPyramidLevels = getImageMipLevels(depthPyramidWidth, depthPyramidHeight);

	VkExtent3D pyramidExtent = {
		static_cast<uint32_t>(depthPyramidWidth),
		static_cast<uint32_t>(depthPyramidHeight),
		1
	};
	//the depth image will be a image with the format we selected and Depth Attachment usage flag
	VkImageCreateInfo pyramidInfo = vkinit::image_create_info(VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, pyramidExtent);

	pyramidInfo.mipLevels = depthPyramidLevels;
	//pyramidInfo.samples = msaaSampleCount;
	//pyramidInfo.initialLayout = VK_IMAGE_LAYOUT_GENERAL;

	//allocate and create the image
	VK_CHECK(vmaCreateImage(_allocator, &pyramidInfo, &dimg_allocinfo, &_depthPyramid._image, &_depthPyramid._allocation, nullptr));

	//build a image-view for the depth image to use for rendering
	VkImageViewCreateInfo priview_info = vkinit::imageview_create_info(VK_FORMAT_R32_SFLOAT, _depthPyramid._image, VK_IMAGE_ASPECT_COLOR_BIT);
	priview_info.subresourceRange.levelCount = depthPyramidLevels;


	VK_CHECK(vkCreateImageView(_device, &priview_info, nullptr, &_depthPyramid._defaultView));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _depthPyramid._defaultView, nullptr);
		vmaDestroyImage(_allocator, _depthPyramid._image, _depthPyramid._allocation);
		
		});

	// One image view per pyarmid levels
	for (int32_t i = 0; i < depthPyramidLevels; ++i)
	{
		VkImageViewCreateInfo level_info = vkinit::imageview_create_info(VK_FORMAT_R32_SFLOAT, _depthPyramid._image, VK_IMAGE_ASPECT_COLOR_BIT);
		level_info.subresourceRange.levelCount = 1;
		level_info.subresourceRange.baseMipLevel = i;

		VkImageView pyramid;
		vkCreateImageView(_device, &level_info, nullptr, &pyramid);
		_mainDeletionQueue.push_function([=]() {
			vkDestroyImageView(_device, pyramid, nullptr);
			
			});
		depthPyramidMips[i] = pyramid;
		assert(depthPyramidMips[i]);
	}
	

	VkSamplerCreateInfo createInfo = {};

	//VkSamplerReductionModeEXT reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN_EXT;
	VkSamplerReductionMode reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;

	createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	createInfo.magFilter = VK_FILTER_LINEAR;
	createInfo.minFilter = VK_FILTER_LINEAR;
	createInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	createInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	createInfo.minLod = 0;
	createInfo.maxLod = 16.f;

	//VkSamplerReductionModeCreateInfoEXT createInfoReduction = { VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT };
	VkSamplerReductionModeCreateInfo createInfoReduction = { VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO };

	if (reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE)
	{
		createInfoReduction.reductionMode = reductionMode;

		createInfo.pNext = &createInfoReduction;
	}

	
	VK_CHECK(vkCreateSampler(_device, &createInfo, 0, &_depthSampler));

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	
	vkCreateSampler(_device, &samplerInfo, nullptr, &_smoothSampler);

	// Shared sampler for cascade depth reads
	// Enable compare feature and set VK_COMPARE_OP_LESS
	VkSamplerCreateInfo shadsamplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
	shadsamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	shadsamplerInfo.compareEnable = true;
	//ref depth value -> shadowmap depth
	//test depth value -> forward pass render image depth
	//ref(0.4) <= test(0.9) -> TURE:obj in shadow , not render
	shadsamplerInfo.compareOp = VK_COMPARE_OP_LESS;//reference value < test value
	vkCreateSampler(_device, &shadsamplerInfo, nullptr, &_shadowSampler);


	//add to deletion queues
	_mainDeletionQueue.push_function([=]() {
		vkDestroySampler(_device, _depthSampler, nullptr);
		vkDestroySampler(_device, _smoothSampler, nullptr);
		vkDestroySampler(_device, _shadowSampler, nullptr);
		});
}

void VulkanEngine::init_forward_renderpass()
{
	//we define an attachment description for our main color image
	//the attachment is loaded as "clear" when renderpass start
	//the attachment is stored when renderpass ends
	//the attachment layout starts as "undefined", and transitions to "Present" so its possible to display it
	//we dont care about stencil, and dont use multisampling

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = _renderFormat;//_swachainImageFormat;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;//PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	// Depth attachment
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;
	//hook the depth attachment into the subpass
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	//1 dependency, which is from "outside" into the subpass. And we can read or write color
	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;//subpass index
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


	//array of 2 attachments, one for the color, and other for depth
	VkAttachmentDescription attachments[2] = { color_attachment,depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	//2 attachments from said array
	render_pass_info.attachmentCount = 2;
	render_pass_info.pAttachments = &attachments[0];
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	//render_pass_info.dependencyCount = 1;
	//render_pass_info.pDependencies = &dependency;

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _renderPass, nullptr);
		});
}


void VulkanEngine::init_copy_renderpass()
{
	//we define an attachment description for our main color image
	//the attachment is loaded as "clear" when renderpass start
	//the attachment is stored when renderpass ends
	//the attachment layout starts as "undefined", and transitions to "Present" so its possible to display it
	//we dont care about stencil, and dont use multisampling

	VkAttachmentDescription color_attachment = {};
	color_attachment.format = _swachainImageFormat;
	color_attachment.samples = msaaSampleCount;
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription colorAttachmentResolve{};
	colorAttachmentResolve.format = _swachainImageFormat;
	colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttachmentResolveRef{};
	colorAttachmentResolveRef.attachment = 1;
	colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachment_ref;	
	subpass.pResolveAttachments = &colorAttachmentResolveRef;

	//1 dependency, which is from "outside" into the subpass. And we can read or write color
	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


	std::array<VkAttachmentDescription, 2> attachments = { color_attachment, colorAttachmentResolve };
	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	//2 attachments from said array
	render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
	render_pass_info.pAttachments = attachments.data();
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;
	//render_pass_info.dependencyCount = 1;
	//render_pass_info.pDependencies = &dependency;

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_copyPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _copyPass, nullptr);
		});
}


void VulkanEngine::init_shadow_renderpass()
{
	VkAttachmentDescription depth_attachment = {};
	// Depth attachment
	depth_attachment.flags = 0;
	depth_attachment.format = _depthFormat;
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	//shader will read depth image(shadow)
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment =0;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

	//hook the depth attachment into the subpass
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	//1 dependency, which is from "outside" into the subpass. And we can read or write color
	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	//2 attachments from said array
	render_pass_info.attachmentCount = 1;
	render_pass_info.pAttachments = &depth_attachment;
	render_pass_info.subpassCount = 1;
	render_pass_info.pSubpasses = &subpass;	

	VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_shadowPass));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(_device, _shadowPass, nullptr);
	});
}

void VulkanEngine::init_framebuffers()
{
	//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
	//render pass rendering target to frame image
	VkFramebufferCreateInfo fwd_info = vkinit::framebuffer_create_info(_renderPass, _windowExtent);
	VkImageView attachments[2];
	attachments[0] = _rawRenderImage._defaultView;
	attachments[1] = _depthImage._defaultView;

	fwd_info.pAttachments = attachments;
	fwd_info.attachmentCount = 2; 
	VK_CHECK(vkCreateFramebuffer(_device, &fwd_info, nullptr, &_forwardFramebuffer));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyFramebuffer(_device, _forwardFramebuffer, nullptr);
		});

	//create the framebuffer for shadow pass	
	VkFramebufferCreateInfo sh_info = vkinit::framebuffer_create_info(_shadowPass, _shadowExtent);
	sh_info.pAttachments = &_shadowImage._defaultView;
	sh_info.attachmentCount = 1;
	VK_CHECK(vkCreateFramebuffer(_device, &sh_info, nullptr, &_shadowFramebuffer));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyFramebuffer(_device, _shadowFramebuffer, nullptr);
		});

	//MailBOX MODEL swapchain image count == 3
	const uint32_t swapchain_imagecount = static_cast<uint32_t>(_swapchainImages.size());
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);
	for (uint32_t i = 0; i < swapchain_imagecount; i++) {

		//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
		VkFramebufferCreateInfo fb_info = vkinit::framebuffer_create_info(_copyPass, _windowExtent);
		std::array<VkImageView, 2> attachments = {
				_colorResourceImage._defaultView,
				_swapchainImageViews[i]
		};
		fb_info.pAttachments = &attachments[0];
		fb_info.attachmentCount = 2;
		VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
			vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
			});
	}
}

void VulkanEngine::init_commands()
{
	//create a command pool for commands submitted to the graphics queue.
	//we also want the pool to allow for resetting of individual command buffers
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	//create command pool per frame(double buffer)
	for (int i = 0; i < FRAME_OVERLAP; i++) {


		VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

		//allocate the main command buffer(every-frame one main cmd) that we will use for rendering
		VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

		VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

		_mainDeletionQueue.push_function([=]() {
			vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
		});

		
	}
	//Profiler register physical device, logical device,graphics queue and command buffer
	_graphicsQueueContext = TracyVkContext(_chosenGPU, _device, _graphicsQueue, _frames[0]._mainCommandBuffer);
	_mainDeletionQueue.push_function([=]() {
		TracyVkDestroy(_graphicsQueueContext);
		std::cout << " desctroy Tracy profiler" << std::endl;
		});
	
	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
	
	//create pool for upload context
	//upload context command pool for immeidately submit
	VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
		});
}

void VulkanEngine::init_sync_structures()
{
	//create syncronization structures
	//one fence to control when the gpu has finished rendering the frame,
	//and 2 semaphores to syncronize rendering with swapchain
	//we want the [fence to start signalled] so we can wait on it on the first frame
	//VK_FENCE_CREATE_SIGNALED_BIT->fence to start signalled
	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);

	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	for (int i = 0; i < FRAME_OVERLAP; i++) {

		VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

		//enqueue the destruction of the fence
		_mainDeletionQueue.push_function([=]() {
			vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
			});


		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));
		VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));

		//enqueue the destruction of semaphores
		_mainDeletionQueue.push_function([=]() {
			vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
			vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
			});
	}


	VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();

	VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext._uploadFence));
	_mainDeletionQueue.push_function([=]() {
		vkDestroyFence(_device, _uploadContext._uploadFence, nullptr);
		});
}


void VulkanEngine::init_pipelines()
{	
	_materialSystem = new vkutil::MaterialSystem();
	//initailize material system before initailize pipeline
	_materialSystem->init(this);
	//_materialSystem->build_default_templates();
	_mainDeletionQueue.push_function([=]() {
		_materialSystem->cleanup();
		delete _materialSystem;
		});
	//fullscreen triangle pipeline for blits
	ShaderEffect* blitEffect = new ShaderEffect();
	blitEffect->add_stage(_shaderCache.get_shader(shader_path("fullscreen.vert.spv")), VK_SHADER_STAGE_VERTEX_BIT);
	blitEffect->add_stage(_shaderCache.get_shader(shader_path("blit.frag.spv")), VK_SHADER_STAGE_FRAGMENT_BIT);
	blitEffect->reflect_layout(_device, nullptr, 0);


	PipelineBuilder pipelineBuilder;

	//input assembly is the configuration for drawing triangle lists, strips, or individual points.
	//we are just going to draw triangle list
	pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	//configure the rasterizer to draw filled triangles
	pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
	pipelineBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;
	//we dont use multisampling, so just run the default one
	pipelineBuilder._multisampling = vkinit::multisampling_state_create_info(msaaSampleCount);

	//a single blend attachment with no blending and writing to RGBA
	pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();


	//default depthtesting
	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_GREATER_OR_EQUAL);

	//build blit pipeline
	pipelineBuilder.setShaders(blitEffect);

	//blit pipeline uses hardcoded triangle so no need for vertex input
	pipelineBuilder.clear_vertex_input();

	pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(false, false, VK_COMPARE_OP_ALWAYS);

	_blitPipeline = pipelineBuilder.build_pipeline(_device, _copyPass);
	_blitLayout = blitEffect->builtLayout;

	_mainDeletionQueue.push_function([=]() {
		//vkDestroyPipeline(_device, meshPipeline, nullptr);
		vkDestroyPipeline(_device, _blitPipeline, nullptr);
		vkDestroyPipelineLayout(_device, _blitLayout, nullptr);
		blitEffect->cleanup(_device);
		delete blitEffect;
	});


	//load the compute shaders and build related pipeline
	load_compute_shader(shader_path("indirect_cull.comp.spv").c_str(), _cullPipeline, _cullLayout);

	load_compute_shader(shader_path("depthReduce.comp.spv").c_str(), _depthReducePipeline, _depthReduceLayout);

	load_compute_shader(shader_path("sparse_upload.comp.spv").c_str(), _sparseUploadPipeline, _sparseUploadLayout);
}

bool VulkanEngine::load_compute_shader(const char* shaderPath, VkPipeline& pipeline, VkPipelineLayout& layout)
{
	ShaderModule computeModule;
	computeModule = *_shaderCache.get_shader(shaderPath);

	//if (!vkutil::load_shader_module(_device, shaderPath, &computeModule))

	//{
	//	std::cout << "Error when building compute shader shader module :" <<shaderPath<< std::endl;
	//	return false;
	//}

	ShaderEffect* computeEffect = new ShaderEffect();;
	computeEffect->add_stage(&computeModule, VK_SHADER_STAGE_COMPUTE_BIT);

	computeEffect->reflect_layout(_device, nullptr, 0);

	ComputePipelineBuilder computeBuilder;
	computeBuilder._pipelineLayout = computeEffect->builtLayout;
	computeBuilder._shaderStage = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeModule.module);


	layout = computeEffect->builtLayout;
	pipeline = computeBuilder.build_pipeline(_device);
	//computeEffect->cleanup(_device);
	//vkDestroyShaderModule(_device, computeModule.module, nullptr);
	//delete computeEffect;
	_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(_device, pipeline, nullptr);
		vkDestroyPipelineLayout(_device, layout, nullptr);
		computeEffect->cleanup(_device);
		delete computeEffect;
	});

	return true;
}


void VulkanEngine::load_meshes()
{
	Mesh triMesh{};
	triMesh.bounds.valid = false;
	//make the array 3 vertices long
	triMesh._vertices.resize(3);

	//vertex positions
	triMesh._vertices[0].position = { 1.f,1.f, 0.0f };
	triMesh._vertices[1].position = { -1.f,1.f, 0.0f };
	triMesh._vertices[2].position = { 0.f,-1.f, 0.0f };

	//vertex colors, all green
	triMesh._vertices[0].color = { 0.f,1.f, 0.0f }; //pure green
	triMesh._vertices[1].color = { 0.f,1.f, 0.0f }; //pure green
	triMesh._vertices[2].color = { 0.f,1.f, 0.0f }; //pure green
	//we dont care about the vertex normals
	upload_mesh(triMesh);
	_meshes["triangle"] = triMesh;
}


void VulkanEngine::load_images()
{
	load_image_to_cache("white", asset_path("white.tx").c_str());
}


bool VulkanEngine::load_image_to_cache(const char* name, const char* path)
{
	ZoneScopedNC("Load Texture", tracy::Color::Yellow);
	Texture newtex;
	//if texture is exit in loaded textures cache, return true;
	if (_loadedTextures.find(name) != _loadedTextures.end()) return true;

	//current texture not exist in loaded texture caches,we should load new texture from assert libaraies
	bool result = vkutil::load_image_from_asset(*this, path, newtex.image);
	//bool result = vkutil::load_image_from_file(*this, path, newtex.image);
	if (!result)
	{
		LOG_ERROR("Error When texture {} at path {}", name, path);
		return false;
	}
	else {
		LOG_SUCCESS("Loaded texture {} at path {}", name, path);
	}
	newtex.imageView = newtex.image._defaultView;
	//VkImageViewCreateInfo imageinfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_UNORM, newtex.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	//imageinfo.subresourceRange.levelCount = newtex.image.mipLevels;
	//vkCreateImageView(_device, &imageinfo, nullptr, &newtex.imageView);

	//new texture should register loaded texture caches
	_loadedTextures[name] = newtex;
	return true;
}

void VulkanEngine::upload_mesh(Mesh& mesh)
{
	ZoneScopedNC("Upload Mesh", tracy::Color::Orange);


	const size_t vertex_buffer_size = mesh._vertices.size() * sizeof(Vertex);
	const size_t index_buffer_size = mesh._indices.size() * sizeof(uint32_t);
	const size_t bufferSize = vertex_buffer_size + index_buffer_size;
	//allocate vertex buffer
	VkBufferCreateInfo vertexBufferInfo = {};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.pNext = nullptr;
	//this is the total size, in bytes, of the buffer we are allocating
	vertexBufferInfo.size = vertex_buffer_size;
	//this buffer is going to be used as a Vertex Buffer
	vertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	//allocate vertex buffer
	VkBufferCreateInfo indexBufferInfo = {};
	indexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	indexBufferInfo.pNext = nullptr;
	//this is the total size, in bytes, of the buffer we are allocating
	indexBufferInfo.size = index_buffer_size;
	//this buffer is going to be used as a Vertex Buffer
	indexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	//let the VMA library know that this data should be writeable by CPU, but also readable by GPU
	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBufferUntyped stagingBuffer;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(_allocator, &vertexBufferInfo, &vmaallocInfo,
		&mesh._vertexBuffer._buffer,
		&mesh._vertexBuffer._allocation,
		nullptr));
	/*_mainDeletionQueue.push_function([=,&mesh]() {
		vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
		});*/
	//copy vertex data
	char* data;
	vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, (void**)&data);

	memcpy(data, mesh._vertices.data(), vertex_buffer_size);

	vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);

	if (index_buffer_size != 0)
	{
		//allocate the buffer
		VK_CHECK(vmaCreateBuffer(_allocator, &indexBufferInfo, &vmaallocInfo,
			&mesh._indexBuffer._buffer,
			&mesh._indexBuffer._allocation,
			nullptr));
		/*_mainDeletionQueue.push_function([=, &mesh]() {
			vmaDestroyBuffer(_allocator, mesh._indexBuffer._buffer, mesh._indexBuffer._allocation);
			});*/
		vmaMapMemory(_allocator, mesh._indexBuffer._allocation, (void**)&data);

		memcpy(data, mesh._indices.data(), index_buffer_size);

		vmaUnmapMemory(_allocator, mesh._indexBuffer._allocation);
	}
	
}

Mesh* VulkanEngine::get_mesh(const std::string& name)
{
	auto it = _meshes.find(name);
	if (it == _meshes.end()) {
		return nullptr;
	}
	else {
		return &(*it).second;
	}
}

void VulkanEngine::init_scene()
{
	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);

	//VkSampler blockySampler;
	//vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);

	samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);

	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	//info.anisotropyEnable = true;
	samplerInfo.mipLodBias = 2;
	samplerInfo.maxLod = 30.f;
	samplerInfo.minLod = 3;
	
	VkSampler smoothSampler;

	VK_CHECK(vkCreateSampler(_device, &samplerInfo, nullptr, &smoothSampler));
	_mainDeletionQueue.push_function([=]() {
		vkDestroySampler(_device, smoothSampler, nullptr);
		});
	//material system add new material data (texture,template index)
	{
		//filling new material data struct and create new material
		vkutil::MaterialData texturedInfo;
		texturedInfo.baseTemplate = "texturedPBR_opaque";
		texturedInfo.parameters = nullptr;

		vkutil::SampledTexture whiteTex;
		whiteTex.sampler = smoothSampler;
		whiteTex.view = _loadedTextures["white"].imageView;

		texturedInfo.textures.push_back(whiteTex);
		//created new material (texture) based on filled material data info
		vkutil::Material* newmat = _materialSystem->build_material("textured", texturedInfo);
	}
	{
		vkutil::MaterialData matinfo;
		matinfo.baseTemplate = "texturedPBR_opaque";
		matinfo.parameters = nullptr;
	
		vkutil::SampledTexture whiteTex;
		whiteTex.sampler = smoothSampler;
		whiteTex.view = _loadedTextures["white"].imageView;

		matinfo.textures.push_back(whiteTex);

		vkutil::Material* newmat = _materialSystem->build_material("default", matinfo);

	}


	//pre load 4 scenes -> load prefab

	////scene obejct 0 flight helmets
	//int dimHelmets =1;
	//for (int x = -dimHelmets; x <= dimHelmets; x++) {
	//	for (int y = -dimHelmets; y <= dimHelmets; y++) {
	//
	//		glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x * 50, 10, y * 50));
	//		glm::mat4 scale = glm::scale(glm::mat4{1.0}, glm::vec3(50));
	//
	//		load_prefab(asset_path("FlightHelmet.pfb").c_str(),(translation * scale));
	//	}
	//}

	//scene object 1 Sponza
	glm::mat4 sponzaMatrix = glm::scale(glm::mat4{ 1.0 }, glm::vec3(1));;

	load_prefab(asset_path("Sponza.pfb").c_str(), sponzaMatrix);

	//scene object 2 TopDownScifi
	/*glm::mat4 unrealFixRotation = glm::rotate(glm::radians(-90.f), glm::vec3{ 1,0,0 });

	load_prefab(asset_path("scifi/TopDownScifi.pfb").c_str(),  glm::translate(glm::vec3{0,20,0}));
	*/
	//scene object 3 polycity
	//int dimcities = 2;
	//for (int x = -dimcities; x <= dimcities; x++) {
	//	for (int y = -dimcities; y <= dimcities; y++) {
	//
	//		glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x * 300, y, y * 300));
	//		glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(10));
	//
	//		
	//		glm::mat4 cityMatrix = translation;// * glm::scale(glm::mat4{ 1.0f }, glm::vec3(.01f));
	//		//load_prefab(asset_path("scifi/TopDownScifi.pfb").c_str(), unrealFixRotation * glm::scale(glm::mat4{ 1.0 }, glm::vec3(.01)));
	//		//load_prefab(asset_path("PolyCity/PolyCity.pfb").c_str(), cityMatrix);
	//		load_prefab(asset_path("CITY/polycity.pfb").c_str(), cityMatrix);
	//	//	load_prefab(asset_path("scifi/TopDownScifi.pfb").c_str(), cityMatrix);
	//	}
	//}

	////scene object 4 WaterBottle
	//int dimHelmets =1;
	//for (int x = -dimHelmets; x <= dimHelmets; x++) {
	//	for (int y = -dimHelmets; y <= dimHelmets; y++) {
	//
	//		glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x * 30, 10, y * 30));
	//		glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(50));
	//
	//		load_prefab(asset_path("WaterBottle.pfb").c_str(),(translation * scale));
	//	}
	//}

	////scene obejct 5 damaged helmets
	//int dimHelmets =1;
	//for (int x = -dimHelmets; x <= dimHelmets; x++) {
	//	for (int y = -dimHelmets; y <= dimHelmets; y++) {
	//
	//		glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x * 50, 10, y * 50));
	//		glm::mat4 scale = glm::scale(glm::mat4{1.0}, glm::vec3(20));
	//
	//		load_prefab(asset_path("DamagedHelmet.pfb").c_str(),(translation * scale));
	//	}
	//}

}


void VulkanEngine::ready_cull_data(RenderScene::MeshPass& pass, VkCommandBuffer cmd)
{
	//copy from the cleared indirect buffer into the one we will use on rendering. This one happens every frame
	if (pass.clearIndirectBuffer._buffer == VK_NULL_HANDLE)
	{
		return;
	}
	VkBufferCopy indirectCopy;
	indirectCopy.dstOffset = 0;
	indirectCopy.size = pass.batches.size() * sizeof(GPUIndirectObject);
	indirectCopy.srcOffset = 0;
	vkCmdCopyBuffer(cmd, pass.clearIndirectBuffer._buffer, pass.drawIndirectBuffer._buffer, 1, &indirectCopy);

	{
		VkBufferMemoryBarrier barrier = vkinit::buffer_barrier(pass.drawIndirectBuffer._buffer, _graphicsQueueFamily);
		barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		cullReadyBarriers.push_back(barrier);
		//vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
	}
}

AllocatedBufferUntyped VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkMemoryPropertyFlags required_flags)
{
	//allocate vertex buffer
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;
	bufferInfo.size = allocSize;

	bufferInfo.usage = usage;


	//let the VMA library know that this data should be writeable by CPU, but also readable by GPU
	VmaAllocationCreateInfo vmaallocInfo = {};
	vmaallocInfo.usage = memoryUsage;
	vmaallocInfo.requiredFlags = required_flags;
	AllocatedBufferUntyped newBuffer;

	//allocate the buffer
	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo,
		&newBuffer._buffer,
		&newBuffer._allocation,
		nullptr));
	newBuffer._size = allocSize;
	return newBuffer;
}


void VulkanEngine::reallocate_buffer(AllocatedBufferUntyped& buffer, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkMemoryPropertyFlags required_flags /*= 0*/)
{
	AllocatedBufferUntyped newBuffer = create_buffer(allocSize, usage, memoryUsage, required_flags);

	get_current_frame()._frameDeletionQueue.push_function([=]() {
		//NOTICE: destroy previous buffer,not current new buffer 
		vmaDestroyBuffer(_allocator, buffer._buffer, buffer._allocation);
	});

	buffer = newBuffer;
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize)
{
	// Calculate required alignment based on minimum device offset alignment
	size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;
	if (minUboAlignment > 0) {
		alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
	}
	return alignedSize;
}


void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
	
	ZoneScopedNC("Inmediate Submit", tracy::Color::White);

	VkCommandBuffer cmd;

	//allocate the default command buffer that we will use for rendering
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1);

	VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &cmd));

	//begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	function(cmd);
	

	VK_CHECK(vkEndCommandBuffer(cmd));

	VkSubmitInfo submit = vkinit::submit_info(&cmd);


	//submit command buffer to the queue and execute it.
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _uploadContext._uploadFence));

	vkWaitForFences(_device, 1, &_uploadContext._uploadFence, true, 9999999999);
	vkResetFences(_device, 1, &_uploadContext._uploadFence);

	vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}


bool VulkanEngine::load_prefab(const char* path, glm::mat4 root)
{
	int rng = rand();
	
	ZoneScopedNC("Load Prefab", tracy::Color::Red);

	auto pf = _prefabCache.find(path);
	if (pf == _prefabCache.end())
	{
		assets::AssetFile file;
		//parse binary file to assery meta file
		bool loaded = assets::load_binaryfile(path, file);

		if (!loaded) {
			LOG_FATAL("Error When loading prefab file at path {}",path);
			return false;
		}
		else {
			LOG_SUCCESS("Prefab {} loaded to cache", path);
		}

		_prefabCache[path] = new assets::PrefabInfo;
		// parse assert meta file get perfab info
		*_prefabCache[path] = assets::read_prefab_info(&file);
	}

	assets::PrefabInfo* prefab = _prefabCache[path];

	//1,fillint prefab world matrix node map
	std::unordered_map<uint64_t, glm::mat4> node_worldmats;

	std::vector<std::pair<uint64_t, glm::mat4>> pending_nodes;
	for (auto& [k, v] : prefab->node_matrices)
	{
		//node_matrices : unordered_map<uin64_t,int>
		//v->int-> matries array index
		glm::mat4 nodematrix{ 1.f };
		//get transform matrix based on node_matries(index:v)
		auto nm = prefab->matrices[v];
		//
		memcpy(&nodematrix, &nm, sizeof(glm::mat4));

		//check if it has parents
		auto matrixIT = prefab->node_parents.find(k);
		if (matrixIT == prefab->node_parents.end()) {
			//add to worldmats
			//no parent node -> add root node-> multiple root matrix
			node_worldmats[k] = root* nodematrix;
		}
		else {
			//enqueue
			pending_nodes.push_back({ k,nodematrix });
		}
	}

	//process pending nodes list until it empties
	while (pending_nodes.size() > 0)
	{
		for (int i = 0; i < pending_nodes.size(); i++)
		{
			//node -> node_world_matrix array index
			uint64_t node = pending_nodes[i].first;
			
			// we should get current node's parent node and fill transform matrix
			//transform matrix = root_matrix * node_matrix0 * node_matrix1 * .... 

			uint64_t parent = prefab->node_parents[node];
			
			//why not find int matrices array perfab? <- root_matrix*node_matrix
			//try to find parent in cache
			auto matrixIT = node_worldmats.find(parent);
			if (matrixIT != node_worldmats.end()) {

				//transform with the parent
				//matrixIT.second = root_matrix * node_matrix(parent_node_matrix)
				glm::mat4 nodematrix = (matrixIT)->second * pending_nodes[i].second;

				node_worldmats[node] = nodematrix;

				//remove from queue, pop last
				pending_nodes[i] = pending_nodes.back();
				pending_nodes.pop_back();
				i--;
			}
		}
		
	}


	//2,filling prefab renderable mesh objects array
	
	std::vector<MeshObject> prefab_renderables;
	prefab_renderables.reserve(prefab->node_meshes.size());

	for (auto& [k, v] : prefab->node_meshes)
	{
		//k-> map.key,v->mesh node(mesh path + material path)
		//load mesh->get mesh and material int caches info by path

		if (v.mesh_path.find("Sky") != std::string::npos) {
			continue;
		}
		//2.1 load meshes
		//determine mesh_path correspond mesh whether exist in mesh caches or not?
		//get mesh on mesh caches by mesh_path of v(node_mesh)
		Mesh* mesh;
		mesh = get_mesh(v.mesh_path.c_str());
		if (!mesh)
		{
			//Not found mesh for this mesh path,
			//so should try load from original assert meta binary file
			//load mesh vertices staging storage Mesh vertice array(not vertex buffer)(CPU)
			Mesh newmesh{};
			newmesh.load_from_meshasset(asset_path(v.mesh_path).c_str());
			
			//Now ,upload mesh vertices from CPU(mesh.vertires array) to GPU(mesh.vertex buffers)
			upload_mesh(newmesh);

			//register new upload mesh to mesh caches
			//
			_meshes[v.mesh_path.c_str()] = newmesh;
			mesh = &_meshes[v.mesh_path.c_str()];
		}

		//we create a smooth sampler first beacause load texture later
		VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_LINEAR);
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		VkSampler smoothSampler;

		//2.2 load material
		//load material(texture,shader effect,pipeline and pipeline layout)
		//determine current material path correspond material whether exist in caches or not?

		VK_CHECK(vkCreateSampler(_device, &samplerInfo, nullptr, &smoothSampler));
		_mainDeletionQueue.push_function([=]() {
			vkDestroySampler(_device, smoothSampler, nullptr);
			});
		auto materialName = v.material_path.c_str();
		vkutil::Material* objectMaterial = _materialSystem->get_material(materialName);
		if (!objectMaterial)
		{
			//Not found material in material caches
			//so try load material from orginal assert meta binary file
			assets::AssetFile materialFile;
			bool loaded = assets::load_binaryfile(asset_path(materialName).c_str(), materialFile);
			
			if (loaded)
			{
				assets::MaterialInfo material = assets::read_material_info(&materialFile);

				auto texture = material.textures["baseColor"];
				if (texture.size() <= 3)
				{
					//texture path error,using default path
					texture = "white.tx";
				}
				//load  texture from texture caches based on texture name and texture path
				loaded = load_image_to_cache(texture.c_str(), asset_path(texture).c_str());
				
				if (loaded)
				{
					//Can load texture(will be sampled) from texture caches
					vkutil::SampledTexture tex;
					tex.view = _loadedTextures[texture].imageView;
					tex.sampler = smoothSampler;//sampler has been created before

					vkutil::MaterialData info;
					info.textures.push_back(tex);
					info.parameters = nullptr;

					//select render pipeline template by material transparency mode(opaque,transparent,masked)
					if (material.transparency == assets::TransparencyMode::Transparent)
					{
						info.baseTemplate = "texturedPBR_transparent";
					}
					else {
						info.baseTemplate = "texturedPBR_opaque";
					}
					//create a new material object based material createInfo
					objectMaterial = _materialSystem->build_material(materialName, info);

					if (!objectMaterial)
					{
						LOG_ERROR("Error When building material {}", v.material_path);
					}
				}
				else
				{
					LOG_ERROR("Error When loading image at {}", v.material_path);
				}
			}
			else
			{
				LOG_ERROR("Error When loading material at path {}", v.material_path);
			}
		}
		
		//2.3 fill Mesh object and add to prefab_renderables array;
		MeshObject loadMeshObejct;
		
		//transparent objects will be invisible
		loadMeshObejct.bDrawForwardPass = true;
		loadMeshObejct.bDrawShadowPass = true;
		

		//find current node correspond world tranform matrix in node_worldmats map
		glm::mat4 nodematrix{ 1.f };
		//nodematrix = node_worldmats[k];
		auto matrixIT = node_worldmats.find(k);
		if (matrixIT != node_worldmats.end()) {
			auto nm = (*matrixIT).second;
			memcpy(&nodematrix, &nm, sizeof(glm::mat4));
		}		
		
		//loadmesh.mesh = get_mesh(v.mesh_path.c_str());
		loadMeshObejct.mesh = mesh;//get mesh from meshs caches
		loadMeshObejct.transformMatrix = nodematrix;
		loadMeshObejct.material = objectMaterial;
		
		//transform module space bound to world space bound
		refresh_renderbounds(&loadMeshObejct);

		//sort key from location
		int32_t lx = int(loadMeshObejct.bounds.origin.x / 10.f);
		int32_t ly = int(loadMeshObejct.bounds.origin.y / 10.f);

		uint32_t key =  uint32_t(std::hash<int32_t>()(lx) ^ std::hash<int32_t>()(ly^1337));

		loadMeshObejct.customSortKey = 0;// rng;// key;
		

		prefab_renderables.push_back(loadMeshObejct);
		//_renderables.push_back(loadmesh);
	}

	_renderScene.register_object_batch(prefab_renderables.data(), static_cast<uint32_t>(prefab_renderables.size()));



	return true;
}


std::string VulkanEngine::asset_path(std::string_view path)
{
	return "D:/VulKan/Vulkan_Engine/vulkan-guide-engine/assets/assets_export/" + std::string(path);
	//return "D:/VulKan/Vulkan_Engine/vulkan-guide-engine/assets/" + std::string(path);
}

std::string VulkanEngine::shader_path(std::string_view path)
{
	return "../../shaders/" + std::string(path);
}

void VulkanEngine::refresh_renderbounds(MeshObject* object)
{
	//dont try to update invalid bounds
	if (!object->mesh->bounds.valid) return;

	RenderBounds originalBounds = object->mesh->bounds;

	//convert bounds to 8 vertices, and transform those
	std::array<glm::vec3, 8> boundsVerts;

	for (int i = 0; i < 8; i++) {
		boundsVerts[i] = originalBounds.origin;
	}

	boundsVerts[0] += originalBounds.extents * glm::vec3(1, 1, 1);
	boundsVerts[1] += originalBounds.extents * glm::vec3(1, 1, -1);
	boundsVerts[2] += originalBounds.extents * glm::vec3(1, -1, 1);
	boundsVerts[3] += originalBounds.extents * glm::vec3(1, -1, -1);
	boundsVerts[4] += originalBounds.extents * glm::vec3(-1, 1, 1);
	boundsVerts[5] += originalBounds.extents * glm::vec3(-1, 1, -1);
	boundsVerts[6] += originalBounds.extents * glm::vec3(-1, -1, 1);
	boundsVerts[7] += originalBounds.extents * glm::vec3(-1, -1, -1);
	
	//re-calculate current coord system (world coord system)bound max coord/min coord
	glm::vec3 min{ std::numeric_limits<float>().max() };
	glm::vec3 max{ -std::numeric_limits<float>().max() };

	glm::mat4 m = object->transformMatrix;

	//transform every vertex, accumulating max/min
	for (int i = 0; i < 8; i++) {
		boundsVerts[i] = m * glm::vec4(boundsVerts[i],1.f);

		min = glm::min(boundsVerts[i], min);
		max = glm::max(boundsVerts[i], max);
	}

	glm::vec3 extents = (max - min) / 2.f;
	glm::vec3 origin = min + extents;

	//calculate scale size
	float max_scale = 0;
	max_scale = std::max( glm::length(glm::vec3(m[0][0], m[0][1], m[0][2])),max_scale);
	max_scale = std::max( glm::length(glm::vec3(m[1][0], m[1][1], m[1][2])),max_scale);
	max_scale = std::max( glm::length(glm::vec3(m[2][0], m[2][1], m[2][2])),max_scale);

	float radius = max_scale * originalBounds.radius;


	object->bounds.extents = extents;
	object->bounds.origin = origin;
	object->bounds.radius = radius;
	object->bounds.valid = true;
}


void VulkanEngine::unmap_buffer(AllocatedBufferUntyped& buffer)
{
	vmaUnmapMemory(_allocator, buffer._allocation);
}

void VulkanEngine::init_descriptors()
{
	_descriptorAllocator = new vkutil::DescriptorAllocator{};
	_descriptorAllocator->init(_device);

	_descriptorLayoutCache = new vkutil::DescriptorLayoutCache{};
	_descriptorLayoutCache->init(_device);

	//create texture binding info
	VkDescriptorSetLayoutBinding textureBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);

	VkDescriptorSetLayoutCreateInfo set3info = {};
	set3info.bindingCount = 1;
	set3info.flags = 0;
	set3info.pNext = nullptr;
	set3info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set3info.pBindings = &textureBind;

	_singleTextureSetLayout = _descriptorLayoutCache->create_descriptor_layout(&set3info);


	const size_t sceneParamBufferSize = FRAME_OVERLAP * pad_uniform_buffer_size(sizeof(GPUSceneData));


	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		_frames[i].dynamicDescriptorAllocator = new vkutil::DescriptorAllocator{};
		_frames[i].dynamicDescriptorAllocator->init(_device);

		//1 megabyte of dynamic data buffer
		auto dynamicDataBuffer = create_buffer(1000000, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		//map dynamic buffer to push buffer 
		_frames[i].dynamicData.init(_allocator, dynamicDataBuffer, _gpuProperties.limits.minUniformBufferOffsetAlignment); 

		//20 megabyte of debug output
		//debugeOutputBuffer->storage indirect draw command,draw object id and draw batches info
		//debugeOutputBuffer -> GPUindirectObject * sizeof(GPUindirectObject)
		_frames[i].debugOutputBuffer = create_buffer(200000000, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
	}
}

void VulkanEngine::init_imgui()
{
	//1: create descriptor pool for IMGUI
	// the size of the pool is very oversize, but its copied from imgui demo itself.
	VkDescriptorPoolSize pool_sizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = 11;// std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	VkDescriptorPool imguiPool;
	VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));


	// 2: initialize imgui library

	//this initializes the core structures of imgui
	ImGui::CreateContext();
	ImGui::GetIO().IniFilename = NULL;

	//this initializes imgui for SDL
	ImGui_ImplSDL2_InitForVulkan(_window);

	//this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = _instance;
	init_info.PhysicalDevice = _chosenGPU;
	init_info.Device = _device;
	init_info.Queue = _graphicsQueue;
	init_info.DescriptorPool = imguiPool;
	init_info.MinImageCount = 3;
	init_info.ImageCount = 3;
	//init_info.MSAASamples = msaaSampleCount;

	ImGui_ImplVulkan_Init(&init_info, _renderPass);

	//creat imgui operation interface
	//execute a gpu command to upload imgui font textures
	//imgui font texture -> imgui operation interface
	immediate_submit([&](VkCommandBuffer cmd) {
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
		});

	//clear font textures from cpu data
	ImGui_ImplVulkan_DestroyFontUploadObjects();

	//add the destroy the imgui created structures
	_mainDeletionQueue.push_function([=]() {

		vkDestroyDescriptorPool(_device, imguiPool, nullptr);
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		});
}

void VulkanEngine::adjust_image_layout() {

	immediate_submit([&](VkCommandBuffer cmd) {
		VkImageSubresourceRange range;
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = depthPyramidLevels;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		VkImageMemoryBarrier imageBarrier_toCompShader = {};
		imageBarrier_toCompShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

		imageBarrier_toCompShader.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier_toCompShader.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageBarrier_toCompShader.image = _depthPyramid._image;
		imageBarrier_toCompShader.subresourceRange = range;

		imageBarrier_toCompShader.srcAccessMask = 0;
		imageBarrier_toCompShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		//barrier the image into the transfer-receive layout
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toCompShader);

		});

}

glm::mat4 DirectionalLight::get_projection()
{
	glm::mat4 projection = glm::orthoLH_ZO(-shadowExtent.x, shadowExtent.x, -shadowExtent.y, shadowExtent.y, -shadowExtent.z, shadowExtent.z);
	return projection;
}

glm::mat4 DirectionalLight::get_view()
{
	glm::vec3 camPos = lightPosition;

	glm::vec3 camFwd = lightDirection;

	glm::mat4 view = glm::lookAt(camPos, camPos + camFwd, glm::vec3(1, 0, 0));
	return view;
}
