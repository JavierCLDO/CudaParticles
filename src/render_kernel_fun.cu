#include "render_kernel_fun.cuh"

#include <device_launch_parameters.h>
#include <curand.h>
#include <thrust/extrema.h>
#include <cuda_runtime.h>

namespace fen
{

__global__ void kernel_render_heat_map(const float* heat_map, uint32_t* pixels, const uint32_t n)
{
	constexpr float mix_threshold = 0.2f;
	constexpr float black_threshold = 0.0f;
    constexpr float r_threshold = 0.20f;
    constexpr float g_threshold = 0.5f;
    constexpr float b_threshold = 0.75f;
	   
    uint8_t r, g, b;

    for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
    {
        // get color
        const float value = heat_map[i];

        if (value < black_threshold)
        {
            continue;
        }

        if (value < r_threshold + mix_threshold)
            r = static_cast<uint8_t>(255.0f * (value / (r_threshold + mix_threshold)));
        else
            r = 255;

        if (value < r_threshold)
            g = 0;
        else if (value < g_threshold + mix_threshold)
            g = static_cast<uint8_t>(255.0f * ((value - r_threshold) / (g_threshold - r_threshold + mix_threshold)));
        else
            g = 255;

        if (value < g_threshold)
            b = 0;
        else if (value < b_threshold + mix_threshold)
            b = static_cast<uint8_t>(255.0f * ((value - g_threshold) / (b_threshold - g_threshold + mix_threshold)));
        else
            b = 255;

        // draw pixel
        pixels[i] = (r << 24) | (g << 16) | (b << 8) | 255;
    }
}

__device__ static float atomicMax(const float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernel_render_entities(const float* positions, const float* radius, const float* heat, const uint32_t n, float* heat_map, const float min_pos_x, const float max_pos_x, const float min_pos_y, const float max_pos_y, const uint32_t window_width, const uint32_t window_height)
{
    const float rel_x = (max_pos_x - min_pos_x) / static_cast<float>(window_width);
    const float rel_y = (max_pos_y - min_pos_y) / static_cast<float>(window_height);

    float p_scale;

	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
        const float& p_x = positions[i];
        const float& p_y = positions[i + n];

        const int p_int_x = int(positions[i]);
        const int p_int_y = int(positions[i + n]);

        // find pixel
        const size_t pos_x = rel_x * (p_x - min_pos_x);
        const size_t pos_y = rel_y * (p_y - min_pos_y);

        // A particle renders '+' 
        const float* ptr = nullptr;
        const float value = heat[i];

        // look top
        p_scale = p_int_y - (p_y - radius[i]);
        if(pos_y > 0 && p_scale > 0.0f)
        {
            ptr = &heat_map[(pos_y - 1) * window_width + pos_x];
            atomicMax(ptr, value * p_scale);
        }

        // look bottom
        p_scale = (p_y + radius[i]) - (p_int_y + 1);
        if (pos_y < window_height - 1 && p_scale > 0.0f)
        {
            ptr = &heat_map[(pos_y + 1) * window_width + pos_x];
            atomicMax(ptr, value * p_scale);
        }

        // look left
        p_scale = p_int_x - (p_x - radius[i]);
        if (pos_x > 0 && p_scale > 0.0f)
        {
            ptr = &heat_map[pos_y * window_width + (pos_x - 1)];
            atomicMax(ptr, value * p_scale);
        }

        // look right
        p_scale = (p_x + radius[i]) - (p_int_x + 1);
        if (pos_x < window_width - 1 && p_scale > 0.0f)
        {
            ptr = &heat_map[pos_y * window_width + (pos_x + 1)];
            atomicMax(ptr, value * p_scale);
        }

        // middle
        ptr = &heat_map[pos_y * window_width + pos_x];
        atomicMax(ptr, value);
	}
}

__global__ void kernel_blur_heat(const float* heat_map, float* heat_map_output, const uint32_t n, const uint32_t window_width, const uint32_t window_height, const uint8_t filter_size)
{
	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
    {
        int x = i % window_width;
        int y = i / window_width;

        float output = 0.0f;
        unsigned hits = 0;

    	for (int ox = -filter_size; ox <= filter_size; ++ox) {
            for (int oy = -filter_size; oy <= filter_size; ++oy) {

                if ((x + ox) >= 0 && (x + ox) < window_width && (y + oy) >= 0 && (y + oy) < window_height) {

                    output += heat_map[(i + ox + (oy * window_width))];
                    hits++;
                }
            }
        }

        heat_map_output[i] = output / hits;
    }
}

__global__ void kernel_add_blur(float* heat_map, const float* heat_map_blurred, const uint32_t n)
{
    for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
    {
        heat_map[i] += heat_map_blurred[i];
    }
}


} // namespace fen
