
#include "json.hpp"
#include "lz4.h"
#include <material_asset.h>
#include <iostream>

assets::MaterialInfo assets::read_material_info(AssetFile* file)
{
	assets::MaterialInfo info;
	char newtype[5] = "MATX";
	bool res = assets::compareType(file->type, newtype);
	if (!res) {
		std::cout << " Input assert meta file type:" << file->type[0] << file->type[1] << file->type[2] << file->type[3]
			<< " ,but current need file type: " << newtype[0] << newtype[1] << newtype[2] << newtype[3] << std::endl;
		throw std::runtime_error("Error will read material info,but input assert meta file type no match!");
	}
	nlohmann::json material_metadata = nlohmann::json::parse(file->json);
	info.baseEffect = material_metadata["baseEffect"];

	//key : textute name, value: texture file path
	for (auto& [key, value] : material_metadata["textures"].items())
	{
		info.textures[key] = value;
	}
	//key : properties name, value: properties file path
	for (auto& [key, value] : material_metadata["customProperties"].items())
	{
		info.customProperties[key] = value;
	}
	//deefault Opaque(²»Í¸Ã÷)
	info.transparency = TransparencyMode::Opaque;
	//get json info based on json properties name(transparency)
	auto it = material_metadata.find("transparency");
	if (it != material_metadata.end())
	{
		std::string val = (*it);
		if (val.compare("transparent") == 0) {
			info.transparency = TransparencyMode::Transparent;
		}
		if (val.compare("masked") == 0) {
			info.transparency = TransparencyMode::Masked;
		}
	}

	return info;
}

assets::AssetFile assets::pack_material(MaterialInfo* info)
{
	nlohmann::json material_metadata;
	material_metadata["baseEffect"] = info->baseEffect;
	material_metadata["textures"] = info->textures;
	material_metadata["customProperties"] = info->customProperties;

	switch (info->transparency)
	{	
	case TransparencyMode::Transparent:
		material_metadata["transparency"] = "transparent";
		break;
	case TransparencyMode::Masked:
		material_metadata["transparency"] = "masked";
		break;
	}

	//core file header
	AssetFile file;
	file.type[0] = 'M';
	file.type[1] = 'A';
	file.type[2] = 'T';
	file.type[3] = 'X';
	//version 
	file.version = 1;

	std::string stringified = material_metadata.dump();
	file.json = stringified;

	return file;
}
