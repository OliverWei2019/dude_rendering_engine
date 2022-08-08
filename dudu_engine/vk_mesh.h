#pragma once

#include <vk_types.h>
#include <vector>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

constexpr bool logMeshUpload = false;


struct VertexInputDescription {
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags = 0;
};



struct Vertex {

	glm::vec3 position;
	//glm::vec3 normal;
	glm::vec<2, uint8_t> oct_normal;//color;
	glm::vec<3, uint8_t> color;
	glm::vec2 uv;
	static VertexInputDescription get_vertex_description();

	void pack_normal(glm::vec3 n);
	void pack_color(glm::vec3 c);
};
//
struct RenderBounds {
	glm::vec3 origin;
	float radius;
	glm::vec3 extents;
	bool valid;
};
struct Mesh {
	std::vector<Vertex> _vertices;
	std::vector<uint32_t> _indices;

	AllocatedBuffer<Vertex> _vertexBuffer;
	AllocatedBuffer<uint32_t> _indexBuffer;

	RenderBounds bounds;

	bool load_from_meshasset(const char* filename);

	template<typename T>
	void fill_vertex_data(std::vector<Vertex>& _vertices, T& unpackedVertices) {
		for (int i = 0; i < _vertices.size(); i++) {

			_vertices[i].position.x = unpackedVertices[i].position[0];
			_vertices[i].position.y = unpackedVertices[i].position[1];
			_vertices[i].position.z = unpackedVertices[i].position[2];

			vec3 normal = vec3(
				unpackedVertices[i].normal[0],
				unpackedVertices[i].normal[1],
				unpackedVertices[i].normal[2]);
			//code 3d normal to 2d normal
			_vertices[i].pack_normal(normal);
			//code RGB [0,1] to RGB[0,255]
			_vertices[i].pack_color(vec3{ unpackedVertices[i].color[0] ,unpackedVertices[i].color[1] ,unpackedVertices[i].color[2] });

			//uv.y = 1-uv.y
			_vertices[i].uv.x = unpackedVertices[i].uv[0];
			_vertices[i].uv.y = unpackedVertices[i].uv[1];
			//_vertices[i].uv.y = 1 - unpackedVertices[i].uv[1];
		}
	}
};