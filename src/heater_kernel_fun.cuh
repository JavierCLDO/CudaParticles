#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fen
{
namespace heater
{
	__global__ void kernel_heat_entities(float* positions, float* heat, const uint32_t n, const float delta);
}
};