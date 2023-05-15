#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fen
{
namespace movement
{
	__global__ void init_positions(float* positions, const uint32_t n, const float max_rad, const float min_x, const float max_x, const float min_y, const float max_y);
	__global__ void init_vel(const float* positions, float* prev_positions, const float delta_x, const float delta_y, const uint32_t n);
	__global__ void kernel_explode_entities(const float* positions, float* delta_mov, float* heat, const uint32_t n, const float e_x, const float e_y, const float dt, const float radius_sq, const float force);
	__global__ void kernel_constrain_entities(float* positions, const float* radius, const uint32_t n, const float min_pos_x, const float max_pos_x, const float min_pos_y, const float max_pos_y);
	__global__ void kernel_move_entities(float* positions, float* prev_positions, float* delta_mov, const float max_rad, const float* heat, const uint32_t n, const float gravity, const float acc_heat, const float delta);
}
};