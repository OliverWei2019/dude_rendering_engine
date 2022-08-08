#pragma once
#include "asset_loader.h"

namespace assets {

	enum class TextureFormat : uint32_t
	{
		Unknown = 0,
		RGBA8
	};
	
	struct PageInfo {
		uint32_t width;
		uint32_t height;
		uint32_t compressedSize;
		uint32_t originalSize;
	};

	struct TextureInfo {
		uint64_t textureSize;
		TextureFormat textureFormat;//RBGA8
		CompressionMode compressionMode;//None, LZ4

		std::string originalFile;//original file path
		std::vector<PageInfo> pages;//pages info
	};

	TextureInfo read_texture_info(AssetFile* file);

	void unpack_texture(TextureInfo* info, const char* sourcebuffer, size_t sourceSize, char* destination);

	void unpack_texture_page(TextureInfo* info, int pageIndex ,char* sourcebuffer, char* destination);

	AssetFile pack_texture(TextureInfo* info, void* pixelData);
}