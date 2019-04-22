#include "solver_core.h"

matrix package_array(fp_type *buf, int nr, int nc) {
	matrix out;
	out.nr = nr; out.nc = nc;
	out.buf = buf;
	return(out);
}

matrix make_transpose(matrix A) {
	matrix At;
	At.nr = A.nc; At.nc = A.nr;
	At.buf = malloc(A.nr*A.nc*sizeof(fp_type));
	
	for(int j = 0; j < At.nc; j++)
		for(int i = 0; i < At.nr; i++)
			At.buf[i + j*At.nr] = A.buf[j + i*A.nr];
	return(At);
}


/* some sample data */
fp_type A1[25] = {1,-2,4,-4,1,0,3,1,3,-2,3,-2,1,0,-2,-1,3,4,-4,5,2,-2,3,-3,2};

fp_type A2[25] = {1,-4,-4,-2,-3,2,-4,4,2,-2,-4,4,3,-2,4,5,2,-2,5,-4,5,-4,3,-1,-4};

fp_type A3[25] = {5,4,0,0,-1,0,3,0,4,-1,-1,4,3,-3,5,5,4,1,-2,4,1,-2,3,3,-2};

fp_type B1[10] = {7.018838345822058, 2.309139437300471, 27.19482089535137, -20.52060908159505, 12.73779688395843, 12.01883834582206, 2.309139437300456, 40.19482089535136, -28.52060908159506, 16.73779688395842};

fp_type B2[10] = {37.73729595182337, -3.185477420200046, 15.14601932774021, 0.6190068729467155, -20.67045691677755, 46.73729595182334, -9.18547742020002, 19.14601932774021, 2.619006872946702, -29.67045691677753};

fp_type B3[10] = {16.32505493461665, 16.27149190104528, 20.0604114155650, 3.275250873447193, 14.37980386633175, 26.32505493461665, 29.27149190104528, 27.06041141556505, 5.275250873447176, 19.37980386633176};

fp_type BB1[10] = {11.092041, -0.044890, 27.072063, -21.102374, 17.126687, 16.150411, -0.052855, 40.153726, -29.120411, 21.107114};

fp_type BB2[10] = {29.178678, -5.897233, 16.238491, 9.044198, -22.051497, 38.201474, -11.937834, 20.320115, 11.035866, -31.077128};

fp_type BB3[10] = {16.943005, 14.876483, 21.202337, 4.002240, 13.106949, 26.979911, 27.990842, 28.266669, 6.025438, 18.154708};

fp_type Zero[10] = {0,0,0,0,0,0,0,0,0,0};
fp_type One[10] = {1,1,1,1,1,1,1,1,1,1};

fp_type sol1[10] = {0, 1, 2, 3, 4, 1, 2, 3, 4, 5};

fp_type sol2[10] = {0.2178959791, 0.9803656239, 1.7252415386, 2.8298458030, 3.5575598197, 1.1621032412, 1.8705815454, 2.6017198436, 3.8028343078, 4.4947524993};

fp_type sol3[10] = {0.2432563551, 1, 1.8509659211, 2.8749500104,  3.7662508550, 1, 1.8564942772, 2.8296609384, 3.9436532085, 4.8862472174};

fp_type sol4[10] = {0, 0, 0, 1.0344824713, 0, 0, 0, 0, 2.2832278481, 1.2207278481};
/*******************/

/* A series of tests for the distributed LS solver.
 * Reference solutions currently assume DOUBLE PRECISION
 */

int main(int argc, char **argv)
{
	int local_rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);

	matrix matA1 = package_array(A1, 5, 5);
	matrix matA2 = package_array(A2, 5, 5);
	matrix matA3 = package_array(A3, 5, 5);
	matrix matA1t = make_transpose(matA1);
	matrix matA2t = make_transpose(matA2);
	matrix matA3t = make_transpose(matA3);
	
	matrix matB1 = package_array(B1, 5, 2);
	matrix matB2 = package_array(B2, 5, 2);
	matrix matB3 = package_array(B3, 5, 2);
	matrix matB1t = make_transpose(matB1);
	matrix matB2t = make_transpose(matB2);
	matrix matB3t = make_transpose(matB3);

	matrix matBB1 = package_array(BB1, 5, 2);
	matrix matBB2 = package_array(BB2, 5, 2);
	matrix matBB3 = package_array(BB3, 5, 2);
	matrix matBB1t = make_transpose(matBB1);
	matrix matBB2t = make_transpose(matBB2);
	matrix matBB3t = make_transpose(matBB3);

	matrix matZero = package_array(Zero, 5, 2);
	matrix matZerot = make_transpose(matZero);
	matrix matOne = package_array(One, 5, 2);
	matrix matOnet = make_transpose(matOne);

	matrix matsol1 = package_array(sol1, 5, 2);
	matrix matsol2 = package_array(sol2, 5, 2);
	matrix matsol3 = package_array(sol3, 5, 2);
	matrix matsol4 = package_array(sol4, 5, 2);

	ADMM_params ahp;
	ahp.rho = 1;
	ahp.primal_stop = 1e-6;
	ahp.dual_stop = 1e-4;
	ahp.max_iter = 5000;

	fp_type sol_precision = 1e-6;

	linear_data p;
	p.xr = 5;
	p.xc = 2;
	p.transa = 'T';
	p.transb = 'N';
	p.transc = 'N';
	p.C = package_array(NULL, 0, 0);

	matrix x_sol;
	solver_signature solver_lambda;
	
	if(argc == 1) {
		if(local_rank == 0)
			printf("No arguments\n");
	}
	else if(strcmp(argv[1], "-jac") == 0) {
		if(local_rank == 0)
			printf("running jt solver\n");
		solver_lambda = jt_ls_solver;
		ahp.termination_test_freq = 5;
		if(local_rank == 0) {
			p.A = matA1t;
			p.B = matB1;
		}
		else if(local_rank == 1) {
			p.A = matA2t;
			p.B = matB2;

		}
		else if(local_rank == 2) {
			p.A = matA3t;
			p.B = matB3;
		}
	}
	else if(strcmp(argv[1], "-cons") == 0) {
		if(local_rank == 0)
			printf("running centralized consensus ADMM\n");
		solver_lambda = consensus_admm_ls;
		ahp.termination_test_freq = 100;
		if(local_rank == 1) {
			p.A = matA1t;
			p.B = matB1;
		}
		else if(local_rank == 2) {
			p.A = matA2t;
			p.B = matB2;
		}
		else if(local_rank == 3) {
			p.A = matA3t;
			p.B = matB3;
		}
	}
	//run first test
	if(local_rank == 0) {
		p.reg_type = L2;
		p.reg_lambda = 0;
		p.Z = matZerot;
	}	
	solver_lambda(MPI_COMM_WORLD, p, ahp, &x_sol);
	
	if(local_rank == 0) {
		if(argc > 2 && strcmp(argv[2], "-printx") == 0)
			print_matrix(x_sol, 0, "x");
		if(frob_norm(1, x_sol, -1, matsol1) < sol_precision)
			printf("Test 1 correct\n");
		else
			printf("Test 1 failed!!\n");
		free(x_sol.buf);
	}

	//second test
	if(local_rank == 0) {
		p.reg_type = L2;
		p.reg_lambda = 10;

	}
	solver_lambda(MPI_COMM_WORLD, p, ahp, &x_sol);
	if(local_rank == 0) {
		if(argc > 2 && strcmp(argv[2], "-printx") == 0)
			print_matrix(x_sol, 0, "x");
		if(frob_norm(1, x_sol, -1, matsol2) < sol_precision)
			printf("Test 2 correct\n");
		else
			printf("Test 2 failed!!\n");
		free(x_sol.buf);
	}
	//third test
	if(local_rank == 0) {
		p.reg_type = L1;
		p.Z = matOnet;
		p.reg_lambda = 25;
	}
	solver_lambda(MPI_COMM_WORLD, p, ahp, &x_sol);
	if(local_rank == 0) {
		if(argc > 2 && strcmp(argv[2], "-printx") == 0)
			print_matrix(x_sol, 0, "x");
		if(frob_norm(1, x_sol, -1, matsol3) < sol_precision)
			printf("Test 3 correct\n");
		else
			printf("Test 3 failed!!\n");
		free(x_sol.buf);
	}
	//fourth test
	if(local_rank == 0) {
		p.reg_type = L1;
		p.Z = matZerot;
		p.reg_lambda = 1000;
	}
	solver_lambda(MPI_COMM_WORLD, p, ahp, &x_sol);
	if(local_rank == 0) {
		if(argc > 2 && strcmp(argv[2], "-printx") == 0)
			print_matrix(x_sol, 0, "x");
		if(frob_norm(1, x_sol, -1, matsol4) < sol_precision)
			printf("test 4 correct\n");
		else
			printf("test 4 failed!!\n");
		free(x_sol.buf);
	}

	free(matA1t.buf);
	free(matA2t.buf);
	free(matA3t.buf);
	free(matB1t.buf);
	free(matB2t.buf);
	free(matB3t.buf);
	free(matBB1t.buf);
	free(matBB2t.buf);
	free(matBB3t.buf);
	free(matZerot.buf);
	free(matOnet.buf);
	
	MPI_Finalize();
}
