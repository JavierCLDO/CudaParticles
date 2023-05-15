#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "singleton.h"
#include "simple_profiler.h"
#include "profiler_steps_enum.h"

#include "SDL.h"
#undef main

namespace fen
{

/**
 * \brief Singleton class that computes the collision of 2D particles using spatial subdivision
 */
class engine : public Singleton<engine>
{
	friend Singleton;
public:

	virtual ~engine() override = default;

	// Singleton requirements
	engine(const engine& other) = delete;
	engine& operator=(const engine& other) = delete;
	engine(engine&& other) = delete;
	engine& operator=(engine&& other) = delete;

	void run();

private:

	/**
	 * \brief Initializes the device vector with random data
	 */
	void init_objects(unsigned long long seed);

	bool init();
	bool manage_input();

	void apply_heat(const double dt);
	void solve_cols();
	void move(const double dt);
	void explode(const double dt);
	void render();

private:

	// Singleton requirements
	engine() = default;

	bool exit_{ false };

	SimpleProfiler<Engine_Execution_Steps::Engine_ALL_, double, std::milli> profiler;

	// Create host vectors
	thrust::device_vector<float> positions;
	thrust::device_vector<float> prev_positions;
	thrust::device_vector<float> delta_mov;
	thrust::device_vector<float> radius;
	thrust::device_vector<float> heat;
	thrust::device_vector<float> heat_map;
	thrust::device_vector<float> heat_map_blurred;
	thrust::device_vector<uint32_t> pixels;
	thrust::host_vector<uint32_t> h_pixels;

	// SDL window
	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Texture* texture;

	std::uint32_t clear_color;
	uint32_t blur_size{ 1 };

	bool b_explode{ false };
	float explode_p_x{};
	float explode_p_y{};
};
}