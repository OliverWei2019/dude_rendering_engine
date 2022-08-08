set(CMAKE_CXX_STANDARD 17)
# Add source to this project's executable.

file(GLOB ENGINE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable (Dudu_Engine ${ENGINE_FILES})


set_property(TARGET Dudu_Engine PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:Dudu_Engine>")

target_include_directories(Dudu_Engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

 
target_compile_definitions(Dudu_Engine PUBLIC TRACY_ENABLE)
target_compile_definitions(Dudu_Engine PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_FORCE_LEFT_HANDED) 


target_precompile_headers(Dudu_Engine PUBLIC "vk_types.h" "<unordered_map>" "<vector>" "<iostream>" "<fstream>" "<string>" )
target_link_libraries(Dudu_Engine vkbootstrap vma glm tinyobjloader imgui stb_image spirv_reflect)

target_link_libraries(Dudu_Engine Vulkan::Vulkan sdl2 assetlib tracy fmt_lib)
