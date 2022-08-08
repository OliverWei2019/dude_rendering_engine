#pragma once
#include <asset_loader.h>


namespace assets {

	//vertex type everything 32 bit float type data
	//include position,normal,color and uv 4 properties
	struct Vertex_f32_PNCV {

		float position[3];//xyz
		float normal[3];
		float color[3];
		float uv[2];//uv: texture sample coord
	};
	//position at 32 bits, normal at 8 bits, color at 8 bits, uvs at 16 bits float
	struct Vertex_P32N8C8V16 {

		float position[3];
		uint8_t normal[3];
		uint8_t color[3];
		float uv[2];
	};

	enum class VertexFormat : uint32_t
	{
		Unknown = 0,
		PNCV_F32, //everything at 32 bits
		P32N8C8V16 //position at 32 bits, normal at 8 bits, color at 8 bits, uvs at 16 bits float
	};

	//mesh bounds include cube and sphere
	struct MeshBounds {
		
		float origin[3];//no(0,0,0)
		float radius;//sphere radius
		float extents[3];//cube bounds
	};


	struct MeshInfo {
		uint64_t vertexBuferSize;
		uint64_t indexBuferSize;
		MeshBounds bounds;
		VertexFormat vertexFormat;
		char indexSize;
		CompressionMode compressionMode;
		std::string originalFile;// original file path
	};

	//transit assert meta file info to mesh info
	//assert meta info {type,version,json,binaryBlob}
	//NOTICE:JSON
	MeshInfo read_mesh_info(AssetFile* file);
	
	//decompression mesh binaryBlob:sourcebuffer,
	//get vertex buffer size based on MeshInfo
	void unpack_mesh(MeshInfo* info, const char* sourcebuffer, size_t sourceSize, char* vertexBufer, char* indexBuffer);
	//compress vertex data and index data to binary blob
	AssetFile pack_mesh(MeshInfo* info, char* vertexData, char* indexData);

	//calculate mesh bounds size( origin,radius and extent)
	MeshBounds calculateBounds(Vertex_f32_PNCV* vertices, size_t count);
}