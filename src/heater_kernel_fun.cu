#include "heater_kernel_fun.cuh"

#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include "parameters.h"

constexpr float WIDTH = MAX_X - MIN_X;
constexpr float H_X = WIDTH / 3;

namespace fen
{
namespace heater
{

__global__ void kernel_heat_entities(float* positions, float* heat, const uint32_t n, const float delta)
{
	for (unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x; i < n; i += (gridDim.x * blockDim.x))
	{
		heat[i] *= 0.9975f;

		heat[i] = fminf(1.0f, fmaxf(0.0f, heat[i]));
	}
}

}

} // namespace fen
