#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fen
{
	__global__ void kernel_render_heat_map(const float* heat_map, uint32_t* pixels, const uint32_t n);
	__global__ void kernel_render_entities(const float* positions, const float* radius, const float* heat, const uint32_t n, float* heat_map, const float min_pos_x, const float max_pos_x, const float min_pos_y, const float max_pos_y, const uint32_t window_width, const uint32_t window_height);
	__global__ void kernel_blur_heat(const float* heat_map, float* heat_map_output, const uint32_t n, const uint32_t window_width, const uint32_t window_height, const uint8_t filter_size);
	__global__ void kernel_add_blur(float* heat_map, const float* heat_map_blurred, const uint32_t n);
};