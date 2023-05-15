#include "engine.h"

#include <thrust/execution_policy.h>

#include "defines.h"

#include "col_solver.cuh"
#include "col_solver_kernel_fun.cuh"
#include "heater_kernel_fun.cuh"
#include "movement_kernel_fun.cuh"
#include "render_kernel_fun.cuh"
#include "parameters.h"

constexpr double DT = 1.0 / TARGET_FPS;
constexpr double DT_MS = DT * 1000;
constexpr int DT_MSi = DT_MS;
constexpr double SUB_DT = DT / SUB_STEPS;

constexpr float HEAT_MIN = 0.0f;
constexpr float HEAT_MAX = 0.1f;


constexpr float DELTA_MAX = MIN_RAD / 2;
constexpr float ACC_DOWN = 0.5f;
constexpr float ACC_HEAT = 4.0f;

constexpr float EXPLODE_R = WINDOW_WIDTH / 12.0f;
constexpr float EXPLODE_R_SQ = EXPLODE_R * EXPLODE_R;
constexpr float EXPLODE_FORCE = 10.0f;

INIT_INSTANCE_STATIC(fen::engine);

// Gets raw pointer from device_vector
#define GET_RAW_PTR(v) thrust::raw_pointer_cast((v).data())

void fen::engine::init_objects(unsigned long long seed) {

	//CUDA_CALL_2(movement::init_positions, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), NUM_OBJECTS, MAX_RAD, MIN_X, MAX_X, MIN_Y, MAX_Y);

	// random generator
	curandGenerator_t generator;

	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, seed);

	curandGenerateUniform(generator, GET_RAW_PTR(positions), positions.size());
	CUDA_CALL_2(kernel_scale, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), WINDOW_WIDTH, MIN_X, positions.size() / 2); // Positions X
	CUDA_CALL_2(kernel_scale, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions) + positions.size() / 2, WINDOW_HEIGHT, MIN_Y, positions.size() / 2); // Positions Y

	curandGenerateUniform(generator, GET_RAW_PTR(radius), radius.size());
	CUDA_CALL_2(kernel_scale, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(radius), MAX_RAD - MIN_RAD, MIN_RAD, radius.size()); // Radius

	curandGenerateUniform(generator, GET_RAW_PTR(heat), radius.size());
	CUDA_CALL_2(kernel_scale, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(heat), HEAT_MAX - HEAT_MIN, HEAT_MIN, heat.size()); // Radius

	thrust::fill(delta_mov.begin(), delta_mov.end(), 0.0f); // Move delta
}


bool fen::engine::init()
{
	//Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	window = SDL_CreateWindow("Particles",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		MAX_X, MAX_Y, SDL_WINDOW_SHOWN);

	if (window == nullptr)
	{
		fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	if (renderer == nullptr)
	{
		SDL_DestroyWindow(window);
		fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR32, SDL_TEXTUREACCESS_STREAMING, MAX_X, MAX_Y);

	if (texture == nullptr)
	{
		SDL_DestroyWindow(window);
		SDL_DestroyTexture(texture);
		fprintf(stderr, "Texture could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	clear_color = 0x10050077; // RGBA

	printf("Num entities: %llu\nBloom effect: %u\n", NUM_OBJECTS, blur_size);

	return true;
}

bool fen::engine::manage_input()
{
	SDL_Event e;

	while (SDL_PollEvent(&e))
	{
		switch (e.type)
		{
		case SDL_QUIT:
			return false;

		case SDL_WINDOWEVENT:

			switch (e.window.event) {
				case SDL_WINDOWEVENT_CLOSE:   // exit game
					return false;

				default:
					break;
			}

			break;

		case SDL_MOUSEBUTTONUP:

			b_explode = true;
			explode_p_x = fminf(MAX_X, fmaxf(MIN_X, e.button.x));
			explode_p_y = fminf(MAX_Y, fmaxf(MIN_Y, e.button.y));
			break;

		case SDL_KEYUP:

			switch (e.key.keysym.sym)
			{
			case SDL_KeyCode::SDLK_KP_PLUS:
				blur_size++;
				printf("bloom: %u\n", blur_size);
				break;
			case SDL_KeyCode::SDLK_KP_MINUS:
				if(blur_size > 0)
					blur_size--;
				printf("bloom: %u\n", blur_size);
				break;
			default:
				break;
			}

			break;
		default:
			break;
		}
	}
	return true;
}


void fen::engine::run()
{
	if(!init())
		return;

	positions = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	prev_positions = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	delta_mov = thrust::device_vector<float>(NUM_OBJECTS * DIM);
	radius = thrust::device_vector<float>(NUM_OBJECTS);
	heat = thrust::device_vector<float>(NUM_OBJECTS);
	heat_map = thrust::device_vector<float>(WINDOW_HEIGHT * WINDOW_WIDTH);
	heat_map_blurred = thrust::device_vector<float>(WINDOW_HEIGHT * WINDOW_WIDTH);
	pixels = thrust::device_vector<uint32_t>(WINDOW_HEIGHT * WINDOW_WIDTH);
	h_pixels = thrust::host_vector<uint32_t>(WINDOW_HEIGHT * WINDOW_WIDTH);

	// Initialize objects using the GPU
	init_objects(0u);

	prev_positions = positions;
	//CUDA_CALL_2(movement::init_vel, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), GET_RAW_PTR(prev_positions), 0.0f, 0.0f, NUM_OBJECTS);
	//CUDA_CALL_2(movement::kernel_constrain_entities, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), GET_RAW_PTR(radius), NUM_OBJECTS, MIN_X, MAX_X, MIN_Y, MAX_Y);

	// Get a solver instance
	auto col_solver = fen::col_solver::Instance();

	// Reset the instance
	col_solver->reset(NUM_OBJECTS, CALCULATE_RAD ? -1.0f : MAX_RAD, MIN_X, MIN_Y, MAX_X, MAX_Y);

	// Run the solver ITERATIONS times 
	while(!exit_) {

		Uint64 start = SDL_GetPerformanceCounter();

		profiler.start_timing<Engine_Execution_Steps::Engine_Input>();
		if (!manage_input())
			break;
		profiler.finish_timing<Engine_Execution_Steps::Engine_Input>();

		profiler.start_timing<Engine_Execution_Steps::Engine_Heat>();
		apply_heat(DT);
		profiler.finish_timing<Engine_Execution_Steps::Engine_Heat>();

		profiler.start_timing<Engine_Execution_Steps::Engine_Move>();
		if (b_explode)
		{
			b_explode = false;
			explode(DT);
		}
		profiler.finish_timing<Engine_Execution_Steps::Engine_Move>();

		for(unsigned i {0}; i < SUB_STEPS; ++i)
		{
			profiler.start_timing<Engine_Execution_Steps::Engine_SolveCols>();
			solve_cols();
			profiler.finish_timing<Engine_Execution_Steps::Engine_SolveCols>();

			profiler.start_timing<Engine_Execution_Steps::Engine_Move>();
			move(SUB_DT);
			profiler.finish_timing<Engine_Execution_Steps::Engine_Move>();
		}

		profiler.start_timing<Engine_Execution_Steps::Engine_Render>();
		render();
		profiler.finish_timing<Engine_Execution_Steps::Engine_Render>();

		Uint64 end = SDL_GetPerformanceCounter();

		const double elapsedMS = (end - start) / (double)SDL_GetPerformanceFrequency() * 1000.0;
		const Uint32 s = elapsedMS < DT_MSi ? floor(DT_MS - elapsedMS) : 0;

		// Cap to 30 FPS
		SDL_Delay(s);

		profiler.next_step();
	}

	// Print the profiling results
	std::cout << '\n';
	col_solver->get_profiler().print_avg_times<Solver_Execution_Steps>(std::cout);
	std::cout << '\n';
	profiler.print_avg_times<Engine_Execution_Steps>(std::cout);

	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);

	SDL_Quit();
}



void fen::engine::apply_heat(const double dt)
{
	CUDA_CALL_2(heater::kernel_heat_entities, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), GET_RAW_PTR(heat), NUM_OBJECTS, dt);
}

void fen::engine::solve_cols()
{
	auto col_solver = fen::col_solver::Instance();

	col_solver->solve_cols(NUM_BLOCKS, NUM_THREADS, positions, radius, delta_mov, heat, NUM_OBJECTS);

	// Add a step to the profiler (to compute the avg times accordingly)
	col_solver->get_profiler().next_step();
}


void fen::engine::move(const double dt)
{

	CUDA_CALL_2(movement::kernel_move_entities, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), GET_RAW_PTR(prev_positions), GET_RAW_PTR(delta_mov), DELTA_MAX, GET_RAW_PTR(heat), NUM_OBJECTS, ACC_DOWN, ACC_HEAT, dt);

	thrust::fill(thrust::device, delta_mov.begin(), delta_mov.end(), 0.0f);

	CUDA_CALL_2(movement::kernel_constrain_entities, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), GET_RAW_PTR(radius), NUM_OBJECTS, MIN_X, MAX_X, MIN_Y, MAX_Y);
}

void fen::engine::explode(const double dt)
{
	CUDA_CALL_2(movement::kernel_explode_entities, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), GET_RAW_PTR(delta_mov), GET_RAW_PTR(heat), NUM_OBJECTS, explode_p_x, explode_p_y, dt, EXPLODE_R_SQ, EXPLODE_FORCE);
}

void fen::engine::render()
{
	thrust::fill(heat_map.begin(), heat_map.end(), 0.0f);
	thrust::fill(pixels.begin(), pixels.end(), clear_color);

	CUDA_CALL_2(fen::kernel_render_entities, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(positions), GET_RAW_PTR(radius), GET_RAW_PTR(heat), NUM_OBJECTS, GET_RAW_PTR(heat_map), MIN_X, MAX_X, MIN_Y, MAX_Y, WINDOW_WIDTH, WINDOW_HEIGHT);

	cudaDeviceSynchronize();

	if (blur_size > 0)
	{
		CUDA_CALL_2(fen::kernel_blur_heat, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(heat_map), GET_RAW_PTR(heat_map_blurred), WINDOW_HEIGHT * WINDOW_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT, blur_size);
		CUDA_CALL_2(fen::kernel_add_blur, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(heat_map), GET_RAW_PTR(heat_map_blurred), WINDOW_HEIGHT * WINDOW_WIDTH);
	}

	CUDA_CALL_2(fen::kernel_render_heat_map, NUM_BLOCKS, NUM_THREADS)(GET_RAW_PTR(heat_map), GET_RAW_PTR(pixels), WINDOW_HEIGHT * WINDOW_WIDTH);

	cudaDeviceSynchronize();

	int pitch = MAX_X * 4;
	void* texturePixels = nullptr;
	SDL_LockTexture(texture, nullptr, &texturePixels, &pitch);

	// Modify texture pixels
	h_pixels = pixels;
	memcpy(texturePixels, h_pixels.data(), h_pixels.size() * 4);

	SDL_UnlockTexture(texture);

	SDL_RenderCopy(renderer, texture, nullptr, nullptr);

	SDL_RenderPresent(renderer);
}
