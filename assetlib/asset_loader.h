#pragma once
#include <vector>
#include <string>

namespace assets {
	//assert meta file structure
	struct AssetFile {
		//type mean: assert type include:mesh texture and material
		char type[4];//MESH TEXT MATX
		int version;
		std::string json;//assert(mesh,texture,material detail info
		std::vector<char> binaryBlob;//compressed data(mesh vertex,texture pxiel...) 
	};

	//availabel compression lib :LZ4 is fastest
	enum class CompressionMode : uint32_t {
		None,
		LZ4
	};
	//save assert meta file to spec path
	bool save_binaryfile(const char* path, const AssetFile& file);
	//read info from binary file path and fill assert meta file
	bool load_binaryfile(const char* path, AssetFile& outputFile);	

	//transit const char* (string) to Compression mode(LZ4,None)
	assets::CompressionMode parse_compression(const char* f);

	bool compareType(const char* type,const char* newtype);
}
