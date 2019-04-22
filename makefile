test_ls: test_ls.c solver_core.h
	mpicc -o test_ls test_ls.c solver_common.c linalg_core.c jac_transpose_solver.c central_consensus.c -lblas -llapack -lm -fopenmp
