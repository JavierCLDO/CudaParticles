#include "movement_kernel_fun.cuh"

#include <device_launch_parameters.h>
#include <thrust/extrema.h>

namespace fen
{
namespace movement
{

__global__ void init_positions(float* positions, const uint32_t n, const float max_rad, const float min_x, const float max_x, const float min_y, const float max_y)
{
	const float d = max_rad * 2;
	const float pitch = (max_x - min_x) * d;

	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
		positions[i] = fminf(max_x, fmaxf(min_x, static_cast<int>((i * d)) % static_cast<int>(pitch)));
		positions[i + n] = max_y - i / (pitch);
	}
}

__global__ void init_vel(const float* positions, float* prev_positions, const float delta_x, const float delta_y, const uint32_t n)
{
	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
		prev_positions[i] = positions[i] - delta_x;
		prev_positions[i + n] = positions[i + n] - delta_y;
	}
}

__global__ void kernel_explode_entities(const float* positions, float* delta_mov, float* heat, const uint32_t n, const float e_x, const float e_y, const float dt, const float radius_sq, const float force)
{
	float dist_sq, dist, delta;
	float d_x, d_y;
	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
		d_x = positions[i] - e_x;
		d_y = positions[i + n] - e_y;
		dist_sq = (d_x * d_x) + (d_y * d_y);

		if(dist_sq < radius_sq)
		{
			dist = sqrtf(dist_sq);
			delta = 1.0f - dist_sq / radius_sq;
			d_x /= dist;
			d_y /= dist;
			delta_mov[i] += d_x * delta * force * dt;
			delta_mov[i + n] += d_y * delta * force * dt;
			heat[i] += 0.5f * delta;
		}
	}
}

__global__ void kernel_constrain_entities(float* positions, const float* radius, const uint32_t n, const float min_pos_x, const float max_pos_x, const float min_pos_y, const float max_pos_y)
{
	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
		// Constrain
		if (positions[i] < min_pos_x + radius[i]) positions[i] = min_pos_x + radius[i];
		else if (positions[i] > max_pos_x - radius[i]) positions[i] = max_pos_x - radius[i];

		if (positions[i + n] < min_pos_y + radius[i]) positions[i + n] = min_pos_y + radius[i];
		else if (positions[i + n] > max_pos_y - radius[i]) positions[i + n] = max_pos_y - radius[i];
	}
}

__global__ void kernel_move_entities(float* positions, float* prev_positions, float* delta_mov, const float max_rad, const float* heat, const uint32_t n, const float gravity, const float acc_heat, const float delta)
{
	float vel_x;
	float vel_y;

	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
		if (delta_mov[i] > max_rad) 
			delta_mov[i] = max_rad;
		else if (delta_mov[i] < -max_rad) 
			delta_mov[i] = -max_rad;
		if (delta_mov[i + n] > max_rad) 
			delta_mov[i + n] = max_rad;
		else if (delta_mov[i + n] < -max_rad) 
			delta_mov[i + n] = -max_rad;

		positions[i] += delta_mov[i];
		positions[i + n] += delta_mov[i + n];

		vel_x = positions[i] - prev_positions[i];
		vel_y = positions[i + n] - prev_positions[i + n];

		prev_positions[i] = positions[i];
		prev_positions[i + n] = positions[i + n];

		positions[i] += vel_x;
		positions[i + n] += vel_y + ((gravity - heat[i] * acc_heat) * delta * delta);
	}
}

}

} // namespace fen
