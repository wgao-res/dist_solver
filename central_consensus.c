#include "solver_core.h"

/* This file implements centralized consensus ADMM.
 * Currently supported:
 * 1) Distributed regularized least squares
 *
 *
 * 1 DISTRIBUTED LEAST SQUARES
 * 	min_x 1/2\|AX - B\|^2 - <C,X>
 * 		 + lambda/2*h(X - Z)
 *   h() may be \|X\|_2 or \|X\|_1
 *
 *   We introduce a consensus variable Y and solve
 *   	min_{x,y} \sum 1/2\|AX - B\|^2 - <C,X>
 *   		+ lambda/2*h(Y-Z)
 *   	s.t 	Y - X for all blocks X
 *
 *   Extensions: consensus ADMM is extremely general.
 *   However owing to lack of closures in C, implementing
 *   the framework of consensus ADMM separately from the
 *   solution of the specific subproblems is not necessarily
 *   possible (while having efficient subroutines for subproblems).
 *   So we will still need to repeat parts of the ADMM framework
 *   when applying consensus to other distributed problems.
 *
 * */
int local_rank = -1;
int nw = -1;

struct worker_vars {
	matrix local_x
		;
	matrix local_dual;
	matrix prev_x;
};
typedef struct worker_vars wvars;

int init_worker_node(matrix X_spec, wvars *out) {
	
	out->local_x.nr = X_spec.nr;
	out->local_x.nc = X_spec.nc;
	int var_size = X_spec.nr * X_spec.nc;

	out->local_x.buf = malloc(var_size*sizeof(fp_type));
	memset(out->local_x.buf, 0, var_size*sizeof(fp_type));
	
	copy_matrix(out->local_x, &out->local_dual);
	copy_matrix(out->local_x, &out->prev_x);

	if(!out->local_x.buf || !out->local_dual.buf || !out->prev_x.buf)
		return(-1);
	return(1);
}

int worker_admm_iteration(preproc_linear_data pp, wvars wv, ADMM_params ahp) {
	/* It is assumed that pp.lhs contains
	 * L, LL^T = A^TA + \rho*I
	 * and pp.rhs contains A^TB
	 * On entry, wv.local_x should contain
	 * the current value of the consensus X.
	 * On exit, wv.local_x will have the 
	 * updated value of x
	 */

	int x_size = wv.local_x.nr * wv.local_x.nc;
	add_scaled_matrix(-1, wv.local_dual, ahp.rho, wv.local_x);
	add_scaled_matrix(1, pp.rhs, 1, wv.local_x);

	//TRTRS params 
	char uplo = 'L';
	char trans = 'N';
	char diag = 'N';
	int N = wv.local_x.nr;
	int nrhs = wv.local_x.nc;
	int lda = pp.lhs.nr;
	int ldb = wv.local_x.nr;
	int info;

	//Solve LY = RHS
	TRTRS_(&uplo, &trans, &diag, &N, &nrhs, pp.lhs.buf, &lda, wv.local_x.buf, &ldb, &info);

	trans = 'T';
	TRTRS_(&uplo, &trans, &diag, &N, &nrhs, pp.lhs.buf, &lda, wv.local_x.buf, &ldb, &info);

	return(1);
}

int center_admm_iteration(linear_data p, preproc_linear_data pp, wvars wv, ADMM_params ahp) {
	/* On entry, wv.local_x should contain
	 * the SUM of the X_i's from each worker.
	 * wv.local_dual should likewise contain the
	 * SUM of Dual_i from each worker.
	 * On exit, local_x will contain the new value
	 * of the consensus X.
	 */
	int is_Zt = FALSE;
	if(p.transz == 'T')
		is_Zt = TRUE;
	if(p.reg_type == L2) {
		fp_type scale = (fp_type)ahp.rho*nw + p.reg_lambda;
		add_scaled_matrix(1/scale, wv.local_dual, ahp.rho/scale, wv.local_x);
		if(p.Z.buf)
			add_scaled_matrix(p.reg_lambda/scale, p.Z, 1, wv.local_x);
	}
	else if(p.reg_type == L1) {
		add_scaled_matrix(1/(ahp.rho*nw), wv.local_dual, 1.0/nw, wv.local_x);
		pointwise_soft_thresholding(wv.local_x, p.Z, wv.local_x, p.reg_lambda/(2*ahp.rho*nw), is_Zt);
	}
	return(1);
}

int consensus_admm_ls(MPI_Comm ls_comm, linear_data p, ADMM_params ahp, matrix *sol)
{
	MPI_Comm_rank(ls_comm, &local_rank);
	MPI_Comm_size(ls_comm, &nw);
       	nw -= 1; //number of worker nodes

	/*******************/

	matrix X_spec;
	X_spec.nr = p.xr; X_spec.nc = p.xc;
	X_spec.buf = NULL;

	wvars wv;
	init_worker_node(X_spec, &wv);

	int x_size = p.xr * p.xc;

	matrix zero;
	zero.nr = p.xr;
	zero.nc = p.xc;
	if(local_rank == 0) {
		zero.buf = malloc(x_size*sizeof(fp_type));
		memset(zero.buf, 0, x_size*sizeof(fp_type));
	}

	preproc_linear_data pp;
	if(local_rank > 0) {
		preprocess_worker_data(p, &pp);
		admm_rho_cholesky(pp.lhs, ahp.rho);
	}

	int is_Zt = FALSE;
	if(p.transz == 'T')
		is_Zt = TRUE;

	/*Begin main ADMM loop*/
	int iter = 0;
	int optimality_tests[2];
	int *primal_test = optimality_tests;
	int *dual_test = optimality_tests + 1;
	*primal_test = FALSE; *dual_test = FALSE;

	while(!(*primal_test && *dual_test) && iter < ahp.max_iter) {
		//ADMM update on each worker for x_i
		if(local_rank > 0)
			worker_admm_iteration(pp, wv, ahp);
		
		//Reduce SUM(x_i) into cv.local_x at center
		if(local_rank == 0)
			MPI_Reduce(zero.buf, wv.local_x.buf, x_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);	
		else
			MPI_Reduce(wv.local_x.buf, NULL, x_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);

		if(local_rank > 0)
			memcpy(wv.prev_x.buf, wv.local_x.buf, x_size*sizeof(fp_type));
		MPI_Barrier(ls_comm);

		//ADMM update on center for x_0
		if(local_rank == 0)
			center_admm_iteration(p, pp, wv, ahp);
		
		MPI_Bcast(wv.local_x.buf, x_size, MPI_FP_TYPE, 0, ls_comm);	
		if(local_rank > 0) {
			add_scaled_matrix(ahp.rho, wv.prev_x, 1, wv.local_dual);
			add_scaled_matrix(-ahp.rho, wv.local_x, 1, wv.local_dual);
		}
		//reduce SUM(dual) into cv.local_dual on center
		if(local_rank == 0)
			MPI_Reduce(zero.buf, wv.local_dual.buf, x_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);
		else
			MPI_Reduce(wv.local_dual.buf, NULL, x_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);


		if(iter % ahp.termination_test_freq == 0) {
			fp_type tmp_zero[2];
			memset(tmp_zero, 0, 2*sizeof(fp_type));
			
			fp_type optimality_test_reduce[2];
			memset(optimality_test_reduce, 0, 2*sizeof(fp_type));
			
			fp_type *local_primal_gap = optimality_test_reduce;
			fp_type *local_dual_gap = optimality_test_reduce+1;

			if(local_rank > 0) {
				//Primal (feasibility) gap
				*local_primal_gap = frob_norm(1, wv.local_x, -1, wv.prev_x);
				matrix grad_x;
				copy_matrix(wv.local_x, &grad_x);

				//local part of dual (feasibility) gap
				//TRMM_
				char side = 'L';
				char uplo = 'L';
				char transa = 'T';
				char diag = 'N';
				int M = wv.local_x.nr;
				int N = wv.local_x.nc;
				fp_type alpha = 1;
				int lda = pp.lhs.nr;
				int ldb = wv.local_x.nr;

				TRMM_(&side, &uplo, &transa, &diag, &M, &N, &alpha, pp.lhs.buf, &lda, grad_x.buf, &ldb);
				
				transa = 'N';
				TRMM_(&side, &uplo, &transa, &diag, &M, &N, &alpha, pp.lhs.buf, &lda, grad_x.buf, &ldb);

				add_scaled_matrix(-ahp.rho, wv.local_x, 1, grad_x);
				add_scaled_matrix(-1, pp.rhs, 1, grad_x);
				*local_dual_gap = frob_norm(1, grad_x, 1, wv.local_dual);
				free(grad_x.buf);
				
				MPI_Reduce(optimality_test_reduce, NULL, 2, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);

			}
			else if(local_rank == 0) {
				MPI_Reduce(tmp_zero, optimality_test_reduce, 2, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);
			}
			
			if(local_rank == 0) {
				fp_type primal_gap = *local_primal_gap;
				fp_type dual_gap = *local_dual_gap;
				if(p.reg_type == L2) {
					matrix grad_cx;
					copy_matrix(wv.local_x, &grad_cx);
					add_scaled_matrix(-p.reg_lambda, p.Z, p.reg_lambda, grad_cx);
					dual_gap += frob_norm(1, grad_cx, -1, wv.local_dual);
					free(grad_cx.buf);
				}
				else if(p.reg_type == L1)
					dual_gap += dist_l1_subgrad(wv.local_dual, wv.local_x, p.Z, p.reg_lambda/2, is_Zt);
				#ifdef _VERBOSE_
				printf("iter: %d dual: %f primal: %f\n", iter, dual_gap, primal_gap);
				#endif
				
				if(primal_gap < ahp.primal_stop)
					*primal_test = TRUE;
				else
					*primal_test = FALSE;
				if(dual_gap < ahp.dual_stop)
					*dual_test = TRUE;
				else
					*dual_test = FALSE;
			}
			MPI_Bcast(optimality_tests, 2, MPI_INT, 0, ls_comm);
			MPI_Barrier(ls_comm);
		}
		iter += 1;
	}

	if(local_rank == 0)
		copy_matrix(wv.local_x, sol);

	free(wv.local_x.buf);
	free(wv.local_dual.buf);
	free(wv.prev_x.buf);
	if(local_rank == 0)
		free(zero.buf);
	if(local_rank > 0) {
		free(pp.lhs.buf);
		free(pp.rhs.buf);
	}
	
	return(1);
}
