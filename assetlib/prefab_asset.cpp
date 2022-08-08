#include "prefab_asset.h"
#include "json.hpp"
#include "lz4.h"
#include <iostream>


assets::PrefabInfo assets::read_prefab_info(AssetFile* file)
{
	PrefabInfo info;
	nlohmann::json prefab_metadata = nlohmann::json::parse(file->json);
	char newtype[5] = "PRFB";
	bool res = assets::compareType(file->type, newtype);
	if (!res) {
		std::cout << " Input assert meta file type:" << file->type[0] << file->type[1] << file->type[2] << file->type[3]
			<< " ,but current need file type: " << newtype[0] << newtype[1] << newtype[2] << newtype[3] << std::endl;
		std::cout << "Error will read prefab info,but input assert meta file type no match!" << std::endl;
	}
	//info.node_matrices = std::unordered_map<uint64_t,int>(prefab_metadata["node_matrices"]) ;
	for (auto pair : prefab_metadata["node_matrices"].items())
	{
		auto value = pair.value();
		auto k = pair.key();
		info.node_matrices[value[0]] = value[1];
	}

	//info.node_names = std::unordered_map<uint64_t, std::string>(prefab_metadata["node_names"]);
	//auto nodenames =;
	for (auto& [key, value]  : prefab_metadata["node_names"].items())
	{
		info.node_names[value[0]] = value[1];
	}

	//info.node_parents = std::unordered_map<uint64_t, uint64_t>(prefab_metadata["node_parents"]);

	for (auto& [key, value] : prefab_metadata["node_parents"].items())
	{

		info.node_parents[value[0]] = value[1];
	}

	std::unordered_map<uint64_t, nlohmann::json> meshnodes = prefab_metadata["node_meshes"];

	for (auto pair : meshnodes) {
		assets::PrefabInfo::NodeMesh node;
		//pair.second -> JSON
		node.mesh_path = pair.second["mesh_path"];
		node.material_path = pair.second["material_path"];

		info.node_meshes[pair.first] = node;
	}


	size_t nmatrices = file->binaryBlob.size() / (sizeof(float) * 16);
	info.matrices.resize(nmatrices);

	memcpy(info.matrices.data(),file->binaryBlob.data(), file->binaryBlob.size());

	return info;
}

assets::AssetFile assets::pack_prefab(const PrefabInfo& info)
{
	//points to matrix array in the blob
	nlohmann::json prefab_metadata;
	//std::unordered_map<uint64_t, int> node_matrices;
	//std::unordered_map<uint64_t, std::string> node_names;
	//std::unordered_map<uint64_t, uint64_t> node_parents;
	prefab_metadata["node_matrices"] = info.node_matrices;
	prefab_metadata["node_names"]    = info.node_names;
	prefab_metadata["node_parents"]  = info.node_parents;

	std::unordered_map<uint64_t, nlohmann::json> meshindex;
	for (auto pair : info.node_meshes)
	{
		nlohmann::json meshnode;
		meshnode["mesh_path"] = pair.second.mesh_path;
		meshnode["material_path"] = pair.second.material_path;
		meshindex[pair.first] = meshnode;
	}

	prefab_metadata["node_meshes"] = meshindex;

	//core file header
	AssetFile file;
	file.type[0] = 'P';
	file.type[1] = 'R';
	file.type[2] = 'F';
	file.type[3] = 'B';
	file.version = 1;

	file.binaryBlob.resize(info.matrices.size() * sizeof(float) * 16);
	memcpy(file.binaryBlob.data(), info.matrices.data(), info.matrices.size() * sizeof(float) * 16);

	std::string stringified = prefab_metadata.dump();
	file.json = stringified;


	return file;
}
