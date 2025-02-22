﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vector>
#include <array>
#include <unordered_map>

#include <vk_descriptors.h>

struct ShaderModule {
	std::vector<uint32_t> code;//storage shader code(spirv)
	VkShaderModule module;
};
namespace vkutil {
	

	//loads a shader module from a spir-v file. Returns false if it errors	
	bool load_shader_module(VkDevice device, const char* filePath, ShaderModule* outShaderModule);

	uint32_t hash_descriptor_layout_info(VkDescriptorSetLayoutCreateInfo* info);
}



class VulkanEngine;
//holds all information for a given shader set for pipeline
//parse ****.spv file(shader)
struct ShaderEffect {

	struct ReflectionOverrides {
		const char* name;
		VkDescriptorType overridenType;
	};

	void add_stage(ShaderModule* shaderModule, VkShaderStageFlagBits stage);

	void reflect_layout(VkDevice device, ReflectionOverrides* overrides, int overrideCount);

	void fill_stages(std::vector<VkPipelineShaderStageCreateInfo>& pipelineStages);
	
	void cleanup(VkDevice device);
	VkPipelineLayout builtLayout;

	struct ReflectedBinding {
		uint32_t set;
		uint32_t binding;
		VkDescriptorType type;
	};
	std::unordered_map<std::string, ReflectedBinding> bindings;
	std::array<VkDescriptorSetLayout, 4> setLayouts;
	std::array<uint32_t, 4> setHashes;
private:
	struct ShaderStage {
		ShaderModule* shaderModule;
		VkShaderStageFlagBits stage;
	};

	std::vector<ShaderStage> stages;
};
//using order:
//set shader->bind buffer->build dptor sets->apply bind to cmd
struct ShaderDescriptorBinder {
	
	struct BufferWriteDescriptor {
		int dstSet;// current dptor set index in pipeline layout
		int dstBinding;//binding point position in current set
		VkDescriptorType descriptorType;
		VkDescriptorBufferInfo bufferInfo;

		uint32_t dynamic_offset;
	};	

	void bind_buffer(const char* name, const VkDescriptorBufferInfo& bufferInfo);

	void bind_dynamic_buffer(const char* name, uint32_t offset,const VkDescriptorBufferInfo& bufferInfo);

	void apply_binds( VkCommandBuffer cmd);

	//void build_sets(VkDevice device, VkDescriptorPool allocator);
	void build_sets(VkDevice device, vkutil::DescriptorAllocator& allocator);

	void set_shader(ShaderEffect* newShader);

	std::array<VkDescriptorSet, 4> cachedDescriptorSets;
private:
	struct DynOffsets {
		std::array<uint32_t, 16> offsets;
		uint32_t count{ 0 };
	};
	std::array<DynOffsets, 4> setOffsets;

	ShaderEffect* shaders{ nullptr };
	std::vector<BufferWriteDescriptor> bufferWrites;
};

class ShaderCache {

public:

	ShaderModule* get_shader(const std::string& path);

	void init(VkDevice device) { _device = device; };
	void cleanup();
private:
	VkDevice _device;
	std::unordered_map<std::string, ShaderModule> module_cache;
};