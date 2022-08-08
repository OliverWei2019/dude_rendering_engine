#include "material_system.h"
#include <vk_initializers.h>
#include <vk_shaders.h>
#include "logger.h"
#include "vk_engine.h"
//build compute pipeline
VkPipeline ComputePipelineBuilder::build_pipeline(VkDevice device)
{
	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

	pipelineInfo.stage = _shaderStage;
	pipelineInfo.layout = _pipelineLayout;


	VkPipeline newPipeline;
	if (vkCreateComputePipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		LOG_FATAL("Failed to build compute pipeline");
		return VK_NULL_HANDLE;
	}
	else
	{
		return newPipeline;
	}
}
// build graphics pipeline
VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass)
{
	//vertex input stage
	_vertexInputInfo = vkinit::vertex_input_state_create_info();
	//connect the pipeline builder vertex input info to the one we get from Vertex
	_vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	_vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)vertexDescription.attributes.size();

	_vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	_vertexInputInfo.vertexBindingDescriptionCount = (uint32_t)vertexDescription.bindings.size();


	//make viewport state from our stored viewport and scissor.
		//at the moment we wont support multiple viewports or scissors
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.pNext = nullptr;
	//should pre-filling viewport and scissor info struct
	viewportState.viewportCount = 1;
	viewportState.pViewports = &_viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &_scissor;

	//setup dummy color blending. We arent using transparent objects yet
	//the blending is just "no blend", but we do write to the color attachment
	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.pNext = nullptr;

	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &_colorBlendAttachment;

	//build the actual pipeline
	//we now use all of the info structs we have been writing into into this one to create the pipeline
	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

	pipelineInfo.stageCount = (uint32_t)_shaderStages.size();
	pipelineInfo.pStages = _shaderStages.data();
	pipelineInfo.pVertexInputState = &_vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &_inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &_rasterizer;
	pipelineInfo.pMultisampleState = &_multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDepthStencilState = &_depthStencil;
	pipelineInfo.layout = _pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;


	std::vector<VkDynamicState> dynamicStates;
	dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
	dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
	dynamicState.pDynamicStates = dynamicStates.data();
	dynamicState.dynamicStateCount = (uint32_t)dynamicStates.size();

	pipelineInfo.pDynamicState = &dynamicState;

	//its easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		LOG_FATAL("Failed to build graphics pipeline");
		return VK_NULL_HANDLE;
	}
	else
	{
		return newPipeline;
	}
}

void PipelineBuilder::clear_vertex_input()
{
	_vertexInputInfo.pVertexAttributeDescriptions = nullptr;
	_vertexInputInfo.vertexAttributeDescriptionCount = 0;

	_vertexInputInfo.pVertexBindingDescriptions = nullptr;
	_vertexInputInfo.vertexBindingDescriptionCount = 0;
}

void PipelineBuilder::setShaders(ShaderEffect* effect)
{
	_shaderStages.clear();
	//get shader stages info
	effect->fill_stages(_shaderStages);
	//build pipeline layout
	_pipelineLayout = effect->builtLayout;
}



void vkutil::MaterialSystem::init(VulkanEngine* owner)
{
	engine = owner;
	build_default_templates();
}

ShaderEffect* build_effect(VulkanEngine* eng,std::string_view vertexShader, std::string_view fragmentShader) {
	ShaderEffect::ReflectionOverrides overrides[] = {
		{"sceneData", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC},
		{"cameraData", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}
	};
	//textured defaultlit shader
	ShaderEffect* effect = new ShaderEffect();
	ShaderModule* vertexShaderModule = eng->_shaderCache.get_shader(VulkanEngine::shader_path(vertexShader));
	effect->add_stage(vertexShaderModule, VK_SHADER_STAGE_VERTEX_BIT);
	if (fragmentShader.size() > 2)
	{
		ShaderModule* fragShaderModule = eng->_shaderCache.get_shader(VulkanEngine::shader_path(fragmentShader));
		effect->add_stage(fragShaderModule, VK_SHADER_STAGE_FRAGMENT_BIT);
	}
	

	effect->reflect_layout(eng->_device, overrides, 2);
	
	return effect; 
}

void vkutil::MaterialSystem::build_default_templates()
{
	//filling two pipeline(shadow,forward) builder
	fill_builders();

	//default shader effects : reflect descriptor
	ShaderEffect* texturedLit = build_effect(engine,  "tri_mesh_ssbo_instanced.vert.spv" ,"textured_lit.frag.spv" );
	ShaderEffect* defaultLit = build_effect(engine, "tri_mesh_ssbo_instanced.vert.spv" , "default_lit.frag.spv" );
	ShaderEffect* opaqueShadowcast = build_effect(engine, "tri_mesh_ssbo_instanced_shadowcast.vert.spv","");

	default_effect_cache = {
		texturedLit,defaultLit,opaqueShadowcast
	};

	//
	ShaderPass* texturedLitPass = build_shader(engine->_renderPass,forwardBuilder, texturedLit);
	ShaderPass* defaultLitPass = build_shader(engine->_renderPass, forwardBuilder, defaultLit);
	ShaderPass* opaqueShadowcastPass = build_shader(engine->_shadowPass,shadowBuilder, opaqueShadowcast);
	default_pass_cache = {
		texturedLitPass,defaultLitPass,opaqueShadowcastPass
	};
	//First effect template: textured opaque PBR
	{
		EffectTemplate defaultTextured;
		defaultTextured.passShaders[MeshpassType::Transparency] = nullptr;
		defaultTextured.passShaders[MeshpassType::DirectionalShadow] = opaqueShadowcastPass;
		defaultTextured.passShaders[MeshpassType::Forward] = texturedLitPass;

		defaultTextured.defaultParameters = nullptr;
		defaultTextured.transparency = assets::TransparencyMode::Opaque;

		templateCache["texturedPBR_opaque"] = defaultTextured;
	}
	//Second effect template: textured transparent PBR
	{
		PipelineBuilder transparentForward = forwardBuilder;

		transparentForward._colorBlendAttachment.blendEnable = VK_TRUE;
		transparentForward._colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		transparentForward._colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		transparentForward._colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;


		//transparentForward._colorBlendAttachment.colorBlendOp = VK_BLEND_OP_OVERLAY_EXT;
		transparentForward._colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
		
		transparentForward._depthStencil.depthWriteEnable = false;

		transparentForward._rasterizer.cullMode = VK_CULL_MODE_NONE;
		//passes
		ShaderPass* transparentLitPass = build_shader(engine->_renderPass, transparentForward, texturedLit);
		default_pass_cache.push_back(transparentLitPass);

		EffectTemplate defaultTextured;
		defaultTextured.passShaders[MeshpassType::Transparency] = transparentLitPass;
		defaultTextured.passShaders[MeshpassType::DirectionalShadow] = nullptr;
		defaultTextured.passShaders[MeshpassType::Forward] = nullptr;

		defaultTextured.defaultParameters = nullptr;
		defaultTextured.transparency = assets::TransparencyMode::Transparent;

		templateCache["texturedPBR_transparent"] = defaultTextured;
	}
	//Third effect template: colored opaque
	{
		EffectTemplate defaultColored;
		
		defaultColored.passShaders[MeshpassType::Transparency] = nullptr;
		defaultColored.passShaders[MeshpassType::DirectionalShadow] = opaqueShadowcastPass;
		defaultColored.passShaders[MeshpassType::Forward] = defaultLitPass;
		defaultColored.defaultParameters = nullptr;
		defaultColored.transparency = assets::TransparencyMode::Opaque;
		templateCache["colored_opaque"] = defaultColored;
	}
	
	//
}

//
//This is not the function used to create the shader effect!
//This function build ShaderPass
//ShaderPass contains shader effect, pipeline and pipeline layout
//pipeline create based on render pass by engine created

vkutil::ShaderPass* vkutil::MaterialSystem::build_shader(VkRenderPass renderPass, PipelineBuilder& builder, ShaderEffect* effect)
{
	ShaderPass* pass = new ShaderPass();

	pass->effect = effect;
	pass->layout = effect->builtLayout;

	PipelineBuilder pipbuilder = builder;

	pipbuilder.setShaders(effect);

	pass->pipeline = pipbuilder.build_pipeline(engine->_device, renderPass);

	return pass;
}


vkutil::Material* vkutil::MaterialSystem::build_material(const std::string& materialName, const MaterialData& info)
{
	Material* mat;
	//search material in the cache first in case its already built
	auto it = materialCache.find(info);
	if (it != materialCache.end())
	{
		mat = (*it).second;
		materials[materialName] = mat;
	}
	else {

		//need to build the material
		Material *newMat = new Material();
		newMat->original = &templateCache[info.baseTemplate];//distribute effect template based on material transparency
		newMat->parameters = info.parameters;
		//not handled yet
		newMat->passSets[MeshpassType::DirectionalShadow] = VK_NULL_HANDLE;
		newMat->textures = info.textures;


	
		auto& db = vkutil::DescriptorBuilder::begin(engine->_descriptorLayoutCache, engine->_descriptorAllocator);

		for (int i = 0; i < info.textures.size(); i++)
		{
			VkDescriptorImageInfo imageBufferInfo;
			imageBufferInfo.sampler = info.textures[i].sampler;
			imageBufferInfo.imageView = info.textures[i].view;
			imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			db.bind_image(i, &imageBufferInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
		}

			
		db.build(newMat->passSets[MeshpassType::Forward]);
		db.build(newMat->passSets[MeshpassType::Transparency]);
		LOG_INFO("Built New Material {}", materialName);
		//add material to cache
		materialCache[info] = (newMat);
		mat = newMat;
		materials[materialName] = mat;
	}

	return mat;
}

vkutil::Material* vkutil::MaterialSystem::get_material(const std::string& materialName)
{
	auto it = materials.find(materialName);
	if (it != materials.end())
	{
		return(*it).second;
	}
	else {
		return nullptr;
	}
}

void vkutil::MaterialSystem::fill_builders()
{
	//filling shadow pipeline builder structure info
	{
		shadowBuilder.vertexDescription = Vertex::get_vertex_description();

		shadowBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
		
		shadowBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
		shadowBuilder._rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;//shadow map 
		shadowBuilder._rasterizer.depthBiasEnable = VK_TRUE;
		
		shadowBuilder._multisampling = vkinit::multisampling_state_create_info(VK_SAMPLE_COUNT_1_BIT);
		shadowBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();

		//default depthtesting
		////reference value < test value -> true : input depth greater reference depth
		shadowBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS);
	}
	//filling forward pipeline builder structure info
	{
		forwardBuilder.vertexDescription = Vertex::get_vertex_description();
		
		forwardBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		
		forwardBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
		forwardBuilder._rasterizer.cullMode = VK_CULL_MODE_NONE;//BACK_BIT;
		
		forwardBuilder._multisampling = vkinit::multisampling_state_create_info(VK_SAMPLE_COUNT_1_BIT);
		
		forwardBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();

		//default depthtesting
		//reference value >= test value -> true : input depth less reference depth
		forwardBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_GREATER_OR_EQUAL);
	}	
}

void vkutil::MaterialSystem::cleanup() {
	/*if (forwardBuilder._pipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(engine->_device, forwardBuilder._pipelineLayout, nullptr);
		std::cout << " destroy forward pipeline layout" << std::endl;
	}
	if (shadowBuilder._pipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(engine->_device, shadowBuilder._pipelineLayout, nullptr);
		std::cout << " destroy shadow pipeline layout" << std::endl;
	}*/
	if (default_pass_cache.size() > 0) {
		uint32_t count = 0;
		uint32_t validCount = 0;
		for (size_t i = 0; i < default_pass_cache.size(); i++) {
			auto effect = default_pass_cache[i];
			if (effect->pipeline) {
				vkDestroyPipeline(engine->_device, effect->pipeline, nullptr);
				validCount++;
			}
		}
		std::cout << " destroy "<<validCount <<" pipelines" << std::endl;
	}
	if (default_effect_cache.size() > 0) {
		uint32_t count = 0;
		uint32_t validCount = 0;
		for (size_t i = 0; i < default_effect_cache.size(); i++) {
			auto effect = default_effect_cache[i];
			if (effect->builtLayout) {
				vkDestroyPipelineLayout(engine->_device, effect->builtLayout, nullptr);
				validCount++;
			}
			effect->cleanup(engine->_device);
			/*if (effect->setLayouts.size() > 0) {
				for (auto dptorSetLayout : effect->setLayouts) {
					vkDestroyDescriptorSetLayout(engine->_device, dptorSetLayout, nullptr);
				}
			}*/
		}
		std::cout << " destroy "<<validCount <<" pipeline layouts " << std::endl;
	}
	/*for (auto pair : templateCache) {
		auto temp = pair.second;
		auto passes = temp.passShaders.data;
		for (size_t i = 0; i < passes.size(); i++) {
			auto pass = passes[i];
			if (pass && pass->pipeline)
			{
				vkDestroyPipeline(engine->_device, pass->pipeline, nullptr);
				pass->pipeline = VK_NULL_HANDLE;
			}
			if (pass && pass->layout) {
				vkDestroyPipelineLayout(engine->_device, pass->layout, nullptr);
				pass->layout = VK_NULL_HANDLE;
			}
			if (pass && pass->effect && pass->effect->setLayouts.size() > 0) {
				pass->effect->cleanup(engine->_device);

			}
		}
	}*/
	/*for (auto material : materials) {
		auto texts = material.second->textures;
		for (size_t i = 0; i < texts.size(); i++) {
			auto text = texts[i];
			if (text.sampler) {
				vkDestroySampler(engine->_device, text.sampler, nullptr);
				text.sampler = VK_NULL_HANDLE;
			}
		}
	}*/
}

bool vkutil::MaterialData::operator==(const MaterialData& other) const
{
	if (other.baseTemplate.compare(baseTemplate) != 0 || other.parameters != parameters || other.textures.size() != textures.size())
	{
		return false;
	}
	else {
		//binary compare textures
		bool comp = memcmp(other.textures.data(), textures.data(), textures.size() * sizeof(textures[0])) == 0;
		return comp;
	}
}

size_t vkutil::MaterialData::hash() const
{
	using std::size_t;
	using std::hash;

	size_t result = hash<std::string>()(baseTemplate);

	for (const auto& b : textures)
	{
		//pack the binding data into a single int64. Not fully correct but its ok
		size_t texture_hash = (std::hash<size_t>()((size_t)b.sampler) << 3) &&(std::hash<size_t>()((size_t)b.view) >> 7);

		//shuffle the packed binding data and xor it with the main hash
		result ^= std::hash<size_t>()(texture_hash);
	}

	return result;
}

