// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

#include <SDL_events.h>
#include <glm/glm.hpp>


struct PlayerCamera {
	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec3 inputAxis;
	//Yaw(俯仰), pitch(偏航), and roll(翻转) rotations
	float pitch{ 0 }; //up-down rotation
	float yaw{ 0 }; //left-right rotation

	bool bSprint = true;
	bool bLocked = false;
	//process hardware input event 
	void process_input_event(SDL_Event* ev);
	//update camera position based delta seconds
	void update_camera(float deltaSeconds);


	glm::mat4 get_view_matrix();
	glm::mat4 get_projection_matrix(bool bReverse = true);
	glm::mat4 get_rotation_matrix();
};