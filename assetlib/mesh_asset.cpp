#include "mesh_asset.h"
#include "json.hpp"
#include "lz4.h"
#include <iostream>

assets::VertexFormat parse_format(const char* f) {

	if (strcmp(f, "PNCV_F32") == 0)
	{
		return assets::VertexFormat::PNCV_F32;
	}
	else if (strcmp(f, "P32N8C8V16") == 0)
	{
		return assets::VertexFormat::P32N8C8V16;
	}
	else
	{
		return assets::VertexFormat::Unknown;
	}
}

assets::MeshInfo assets::read_mesh_info(AssetFile* file)
{
	MeshInfo info;
	char newtype[5] = "MESH";
	bool res = assets::compareType(file->type, newtype);
	if (!res) {
		std::cout << " Input assert meta file type:" << file->type[0] << file->type[1] << file->type[2] << file->type[3]
			<< " ,but current need file type: " << newtype[0] << newtype[1] << newtype[2] << newtype[3] << std::endl;
		throw std::runtime_error("Error will read mesh info,but input assert meta file type no match!");
	}
	nlohmann::json metadata = nlohmann::json::parse(file->json);
	
	info.vertexBuferSize = metadata["vertex_buffer_size"];		
	info.indexBuferSize = metadata["index_buffer_size"];
	info.indexSize = (uint8_t) metadata["index_size"];
	info.originalFile = metadata["original_file"];

	std::string compressionString = metadata["compression"];
	info.compressionMode = parse_compression(compressionString.c_str());

	std::vector<float> boundsData;
	boundsData.reserve(7);
	//get data based on vector<float>
	boundsData = metadata["bounds"].get<std::vector<float>>();
	//origin(x,y,z) = (data[0],data[1],data[2])
	info.bounds.origin[0] = boundsData[0];
	info.bounds.origin[1] = boundsData[1];
	info.bounds.origin[2] = boundsData[2];

	info.bounds.radius = boundsData[3];
	//extent(x,y,z)
	info.bounds.extents[0] = boundsData[4];
	info.bounds.extents[1] = boundsData[5];
	info.bounds.extents[2] = boundsData[6];
	//transit string data to enum class 
	std::string vertexFormat = metadata["vertex_format"];
	info.vertexFormat = parse_format(vertexFormat.c_str());
    return info;
}

void assets::unpack_mesh(MeshInfo* info, const char* sourcebuffer, size_t sourceSize, char* vertexBufer, char* indexBuffer)
{
	//decompressing into staging temporal vector. 
	//TODO: streaming decompress directly on the buffers
	std::vector<char> decompressedBuffer;
	decompressedBuffer.resize(info->vertexBuferSize + info->indexBuferSize);

	LZ4_decompress_safe(sourcebuffer, decompressedBuffer.data(), static_cast<int>(sourceSize), static_cast<int>(decompressedBuffer.size()));

	//copy vertex buffer
	memcpy(vertexBufer, decompressedBuffer.data(), info->vertexBuferSize);

	//copy index buffer
	memcpy(indexBuffer, decompressedBuffer.data() + info->vertexBuferSize, info->indexBuferSize);
}

assets::AssetFile assets::pack_mesh(MeshInfo* info, char* vertexData, char* indexData)
{
    AssetFile file;
	file.type[0] = 'M';
	file.type[1] = 'E';
	file.type[2] = 'S';
	file.type[3] = 'H';
	file.version = 1;

	nlohmann::json metadata;
	if (info->vertexFormat == VertexFormat::P32N8C8V16) {
		metadata["vertex_format"] = "P32N8C8V16";
	}
	else if (info->vertexFormat == VertexFormat::PNCV_F32)
	{
		metadata["vertex_format"] = "PNCV_F32";
	}
	metadata["vertex_buffer_size"] = info->vertexBuferSize;
	metadata["index_buffer_size"] = info->indexBuferSize;
	metadata["index_size"] = info->indexSize;
	metadata["original_file"] = info->originalFile;

	std::vector<float> boundsData;
	boundsData.resize(7);

	boundsData[0] = info->bounds.origin[0];
	boundsData[1] = info->bounds.origin[1];
	boundsData[2] = info->bounds.origin[2];

	boundsData[3] = info->bounds.radius;

	boundsData[4] = info->bounds.extents[0];
	boundsData[5] = info->bounds.extents[1];
	boundsData[6] = info->bounds.extents[2];

	metadata["bounds"] = boundsData;

	size_t fullsize = info->vertexBuferSize + info->indexBuferSize;

	std::vector<char> merged_buffer;
	merged_buffer.resize(fullsize);

	//copy vertex buffer
	memcpy(merged_buffer.data(), vertexData, info->vertexBuferSize);

	//copy index buffer
	memcpy(merged_buffer.data() + info->vertexBuferSize, indexData, info->indexBuferSize);


	//compress buffer and copy it into the file struct
	size_t compressStaging = LZ4_compressBound(static_cast<int>(fullsize));

	file.binaryBlob.resize(compressStaging);

	int compressedSize = LZ4_compress_default(merged_buffer.data(), file.binaryBlob.data(), static_cast<int>(merged_buffer.size()), static_cast<int>(compressStaging));
	file.binaryBlob.resize(compressedSize);

	std::cout <<"Compression rate:"<<(float)(compressedSize / compressStaging) << std::endl;

	metadata["compression"] = "LZ4";

	file.json = metadata.dump();

	return file;
}

assets::MeshBounds assets::calculateBounds(Vertex_f32_PNCV* vertices, size_t count)
{
	MeshBounds bounds;

	float minSize[3] = { std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max() };
	float maxSize[3] = { std::numeric_limits<float>::min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::min() };

	//get max positon location and min position location
	for (int i = 0; i < count; i++) {
		minSize[0] = std::min(minSize[0], vertices[i].position[0]);
		minSize[1] = std::min(minSize[1], vertices[i].position[1]);
		minSize[2] = std::min(minSize[2], vertices[i].position[2]);

		maxSize[0] = std::max(maxSize[0], vertices[i].position[0]);
		maxSize[1] = std::max(maxSize[1], vertices[i].position[1]);
		maxSize[2] = std::max(maxSize[2], vertices[i].position[2]);
	}

	bounds.extents[0] = (maxSize[0] - minSize[0]) / 2.0f;
	bounds.extents[1] = (maxSize[1] - minSize[1]) / 2.0f;
	bounds.extents[2] = (maxSize[2] - minSize[2]) / 2.0f;

	bounds.origin[0] = bounds.extents[0] + minSize[0];
	bounds.origin[1] = bounds.extents[1] + minSize[1];
	bounds.origin[2] = bounds.extents[2] + minSize[2];

	//go through the vertices again to calculate the exact bounding sphere radius
	float r2 = 0;
	for (int i = 0; i < count; i++) {

		float offset[3];
		offset[0] = vertices[i].position[0] - bounds.origin[0];
		offset[1] = vertices[i].position[1] - bounds.origin[1];
		offset[2] = vertices[i].position[2] - bounds.origin[2];

		//pithagoras
		//distance = x^2+y^2+z^2
		float distance = offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2];
		r2 = std::max(r2, distance);
	}

	bounds.radius = std::sqrt(r2);

	return bounds;
}
