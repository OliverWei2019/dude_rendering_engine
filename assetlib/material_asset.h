#pragma once
#include <asset_loader.h>


namespace assets {
	//Transparency Materieal properties
	enum class TransparencyMode:uint8_t {
		Opaque,//²»Í¸Ã÷
		Transparent,//Í¸Ã÷
		Masked//ÃÉÆ¤
	};

	struct MaterialInfo {
		std::string baseEffect;
		std::unordered_map<std::string, std::string> textures; //name -> path
		std::unordered_map<std::string, std::string> customProperties;// properties defined by gltf file
		TransparencyMode transparency;
	};
	//read material info from assert meta file 
	//AssertFile{type(MATX),verison,json,binaryBlob}
	//1,determin type and version
	//2,parse json get info
	MaterialInfo read_material_info(AssetFile* file);
	
	//fill assert meta file JSON based on material info
	AssetFile pack_material(MaterialInfo* info);
}