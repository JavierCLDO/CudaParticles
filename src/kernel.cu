#include "col_solver.cuh"
#include "engine.h"


int main(void)
{
	// Create a solver instance
	fen::col_solver::CreateInstance();
	fen::engine::CreateInstance();
	
	fen::engine::Instance()->run();

    return 0;
}
