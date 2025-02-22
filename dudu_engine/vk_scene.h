﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vk_scene.h>
#include <vk_mesh.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <array>
#include <vector>
#include <unordered_map>
#include "material_system.h"

template<typename T>
struct Handle {
	uint32_t handle;
};

struct MeshObject;
struct Mesh;
struct GPUObjectData;
namespace vkutil { struct Material; }
namespace vkutil { struct ShaderPass; }

//indirected call object 
struct GPUIndirectObject {
	//command is a Structure specifying a indexed indirect drawing command
	VkDrawIndexedIndirectCommand command;
	uint32_t objectID;
	uint32_t batchID;
};

struct DrawMesh {
	uint32_t firstVertex;
	uint32_t firstIndex;
	uint32_t indexCount;
	uint32_t vertexCount;
	bool isMerged;//merged flag determine whether batch merged or not

	Mesh* original;//
};



struct RenderObject {

	Handle<DrawMesh> meshID;
	Handle<vkutil::Material> material;

	uint32_t updateIndex;
	uint32_t customSortKey{0};
	//3 render pass array 
	vkutil::PerPassData<int32_t> passIndices;

	glm::mat4 transformMatrix;

	RenderBounds bounds;
};

struct GPUInstance {
	uint32_t objectID;
	uint32_t batchID;
};


class RenderScene {
public:
	// material in pass
	struct PassMaterial {
		VkDescriptorSet materialSet;//descriptor set
		vkutil::ShaderPass* shaderPass;//inlcude : effect,pipeline,pipeline layout

		bool operator==(const PassMaterial& other) const
		{
			return materialSet == other.materialSet && shaderPass == other.shaderPass;
		}
	};
	//
	struct PassObject {
		PassMaterial material;
		Handle<DrawMesh> meshID;
		Handle<RenderObject> original;
		int32_t builtbatch;
		uint32_t customKey;
	};
	//
	struct RenderBatch {
		Handle<PassObject> object;
		uint64_t sortKey;

		bool operator==(const RenderBatch& other) const
		{
			return object.handle == other.object.handle && sortKey == other.sortKey;
		}
	};
	struct IndirectBatch {
		Handle<DrawMesh> meshID;
		PassMaterial material;
		uint32_t first;
		uint32_t count;
	};
	
	struct Multibatch {
		uint32_t first;
		uint32_t count;
	};
	//MeshPass is not Vulkan render pass,
	//MeshPass contains the resource index and resource layout required for the specified render pass
	//resource index mean renderable objects index for cache,such as mesh,material
	//resource layout mean allocate batches by resource index
	struct MeshPass {
		// final draw-indirect segments
		std::vector<RenderScene::Multibatch> multibatches;
		//Batches is an array of DrawIndirect data, 
		//each of them covering a range on the flat-batches array.
		//Extract multile objects with same feature in flat_batches,
		//batch a element to storage batches array
		//IndirectBatch = {meshid,material,first,count}
		//batches: |b0|b1|b2|b3|b4|
		//meshid:  |1 |2 |3 |2 |3 | 
		//material:|1 |1 |1 |2 |2 | 
		//first：  |22|00|02|16|29|
		//count:   |1 |2 |1 |1 |3 |
		//NOTICE: first->array index,count->using same mesh and material object number
		std::vector<RenderScene::IndirectBatch> batches;

		std::vector<Handle<RenderObject>> unbatchedObjects;
		//Flat_batches is the individual non-instanced draws for every object in the pass
		//RenderBatch = {obj,sortkey}
		//flat_bactches:|obj0|obj1|obj2|obj2|obj3| ...   
		//				|key0|key1|key2|key2|key3| ...
		std::vector<RenderScene::RenderBatch> flat_batches;

		std::vector<PassObject> objects;

		std::vector<Handle<PassObject>> reusableObjects;

		std::vector<Handle<PassObject>> objectsToDelete;

		
		AllocatedBuffer<uint32_t> compactedInstanceBuffer;
		AllocatedBuffer<GPUInstance> passObjectsBuffer;

		AllocatedBuffer<GPUIndirectObject> drawIndirectBuffer;
		AllocatedBuffer<GPUIndirectObject> clearIndirectBuffer;

		PassObject* get(Handle<PassObject> handle);

		MeshpassType type;
		void cleanup(class VulkanEngine* engine);

		bool needsIndirectRefresh = true;
		bool needsInstanceRefresh = true;
	};

	void init();
	/// <summary>
	/// MeshObject{
	///		Mesh* msh;
	///		Material* material;
	///		uint32t customSortKey;
	///		mat4 transformMatrix;
	///		RenderBounds bounds;
	///		uint32_t bDrawForwardPass : 1;
	///		uint32_t bDrawShadowPass : 1;
	/// </summary>
	/// <param name="object"></param>
	/// <returns></returns>
	Handle<RenderObject> register_object(MeshObject* object);

	void register_object_batch(MeshObject* first, uint32_t count);

	void update_transform(Handle<RenderObject> objectID,const glm::mat4 &localToWorld);
	void update_object(Handle<RenderObject> objectID);
	
	void fill_objectData(GPUObjectData* data);
	void fill_indirectArray(GPUIndirectObject* data, MeshPass& pass);
	void fill_instancesArray(GPUInstance* data, MeshPass& pass);

	void write_object(GPUObjectData* target, Handle<RenderObject> objectID);
	
	void clear_dirty_objects();

	void build_batches();

	void merge_meshes(class VulkanEngine* engine);

	void refresh_pass(MeshPass* pass);

	void flush(class VulkanEngine* engine);

	void build_indirect_batches(MeshPass* pass, std::vector<IndirectBatch>& outbatches, std::vector<RenderScene::RenderBatch>& inobjects);
	RenderObject* get_object(Handle<RenderObject> objectID);
	DrawMesh* get_mesh(Handle<DrawMesh> objectID);

	vkutil::Material *get_material(Handle<vkutil::Material> objectID);

	std::vector<RenderObject> renderables;
	std::vector<DrawMesh> meshes;
	std::vector<vkutil::Material*> materials;

	std::vector<Handle<RenderObject>> dirtyObjects;

	MeshPass* get_mesh_pass(MeshpassType name);

	//_forwardPass is not a real vulkan renderpass 
	MeshPass _forwardPass;
	//_transparentForwardPass is not a real vulkan renderpass 
	MeshPass _transparentForwardPass;
	//_shadowPass is not a real vulkan renderpass 
	MeshPass _shadowPass;

	std::unordered_map<vkutil::Material*, Handle<vkutil::Material>> materialConvert;
	std::unordered_map<Mesh*, Handle<DrawMesh>> meshConvert;

	Handle<vkutil::Material> getMaterialHandle(vkutil::Material* m);
	Handle<DrawMesh> getMeshHandle(Mesh* m);
	

	AllocatedBuffer<Vertex> mergedVertexBuffer;
	AllocatedBuffer<uint32_t> mergedIndexBuffer;

	AllocatedBuffer<GPUObjectData> objectDataBuffer;
};

