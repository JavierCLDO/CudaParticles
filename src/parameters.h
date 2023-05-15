#pragma once

// Default values for parameters
#ifdef NDEBUG
constexpr unsigned int NUM_BLOCKS = 256;
constexpr unsigned int NUM_THREADS = 512;
constexpr unsigned int SUB_STEPS = 4;
constexpr float MIN_RAD = 0.4f;
constexpr float MAX_RAD = 0.8f;
constexpr bool CALCULATE_RAD = false;

constexpr float MIN_X = 0.0f;
constexpr float MIN_Y = 0.0f;
constexpr float MAX_X = 1024.0f;
constexpr float MAX_Y = 1024.0f;

constexpr uint32_t WINDOW_WIDTH = MAX_X - MIN_X;
constexpr uint32_t WINDOW_HEIGHT = MAX_Y - MIN_Y;

constexpr uint32_t TOTAL_PIXELS = WINDOW_HEIGHT * WINDOW_WIDTH;

constexpr size_t MAX_NUM_OBJECTS = TOTAL_PIXELS / ((MIN_RAD + (MAX_RAD - MIN_RAD) / 2) * 2);
constexpr size_t NUM_OBJECTS = MAX_NUM_OBJECTS * 0.75;

constexpr double TARGET_FPS = 30.0;
#else
constexpr unsigned int NUM_BLOCKS = 256;
constexpr unsigned int NUM_THREADS = 512;
constexpr unsigned int SUB_STEPS = 1;
constexpr float MIN_RAD = 0.5f;
constexpr float MAX_RAD = 1.0f;
constexpr bool CALCULATE_RAD = false;

constexpr float MIN_X = 0.0f;
constexpr float MIN_Y = 0.0f;
constexpr float MAX_X = 256.0f;
constexpr float MAX_Y = 256.0f;

constexpr uint32_t WINDOW_WIDTH = MAX_X - MIN_X;
constexpr uint32_t WINDOW_HEIGHT = MAX_Y - MIN_Y;

constexpr uint32_t TOTAL_PIXELS = WINDOW_HEIGHT * WINDOW_WIDTH;

constexpr size_t MAX_NUM_OBJECTS = TOTAL_PIXELS / ((MIN_RAD + (MAX_RAD - MIN_RAD) / 2) * 2);
constexpr size_t NUM_OBJECTS = MAX_NUM_OBJECTS * 0.75;

constexpr double TARGET_FPS = 30.0;
#endif