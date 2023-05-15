#pragma once

/**
 * \brief Profiler steps
 */
enum Solver_Execution_Steps : unsigned {
	Cells_Init, Sort, Cols_Init, Cols_Resolve,
	ALL_
};


/**
 * \brief prefix ++ operator for the enum
 */
inline Solver_Execution_Steps& operator++(Solver_Execution_Steps& orig)
{
	orig = static_cast<Solver_Execution_Steps>(orig + 1); // static_cast required because enum + int -> int
	return orig;
}

/**
 * \brief Prints the enum name to the output stream
 */
inline std::ostream& operator<<(std::ostream& os, const Solver_Execution_Steps& e)
{
	switch (e)
	{
	case Cells_Init:	os << "Cells_Init"; break;
	case Sort:			os << "Sort"; break;
	case Cols_Init:		os << "Cols_Init"; break;
	case Cols_Resolve:	os << "Cols_Resolve"; break;
	case ALL_: break;
	}
	return os;
}

/**
 * \brief Profiler steps
 */
enum Engine_Execution_Steps : unsigned {
	Engine_Input, Engine_SolveCols, Engine_Heat, Engine_Move, Engine_Render,
	Engine_ALL_
};


/**
 * \brief prefix ++ operator for the enum
 */
inline Engine_Execution_Steps& operator++(Engine_Execution_Steps& orig)
{
	orig = static_cast<Engine_Execution_Steps>(orig + 1); // static_cast required because enum + int -> int
	return orig;
}

/**
 * \brief Prints the enum name to the output stream
 */
inline std::ostream& operator<<(std::ostream& os, const Engine_Execution_Steps& e)
{
	switch (e)
	{
	case Engine_Input:			os << "Engine_Input"; break;
	case Engine_SolveCols:		os << "Engine_SolveCols"; break;
	case Engine_Heat:			os << "Engine_Heat"; break;
	case Engine_Move:			os << "Engine_Move"; break;
	case Engine_Render:			os << "Engine_Render"; break;
	case Engine_ALL_: break;
	}
	return os;
}