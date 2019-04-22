#include "solver_core.h"

/* This file implements Jacobian transpose ADMM.
 * Currently supported:
 * 1) Distributed regularized least squares
 *
 *
 * 1 DISTRIBUTED LEAST SQUARES
 * 	min_x 1/2\|AX - B\|^2 - <C,X>
 * 		 + lambda/2*h(X - Z)
 *   h() may be \|X\|_2 or \|X\|_1
 *
 *   The algorithm used computes A^TA and A^TB
 *   locally at each distributed worker, and sums
 *   these in the center node. The center then computes
 *   the solution. This is either analytic (for L2) or 
 *   requires running ADMM (for L1).
 *
 *   Extensions: more general problems that can be solved
 *   with unwrapped ADMM.
 *
 * */

int jt_ls_solver(MPI_Comm ls_comm, linear_data p, ADMM_params ahp, matrix *sol)
{
	
	int local_rank = -1;
	int nw = -1;

	MPI_Comm_rank(ls_comm, &local_rank);
	MPI_Comm_size(ls_comm, &nw);
	
	int x_size = p.xr*p.xc;

	/* We store several matrices at the center:
	 * center_lhs, center_rhs, center_ATA, center_ATB
	 * Since the source and target of REDUCE cannot be
	 * the same location, the targets will be center_{lhs,rhs}
	 * center_{ATA, ATB} will store either the preprocessed
	 * data at the center, or the zero matrix.
	 *
	 * The solution will eventually be saved in center_rhs.
	 */
	matrix center_ATA;
	center_ATA.nr = p.xr; center_ATA.nc = p.xr;
	matrix center_ATB;
	center_ATB.nr = p.xr; center_ATB.nc = p.xc;

	int ATA_size = p.xr*p.xr;
	int ATB_size = p.xr*p.xc;

	matrix center_lhs;
	center_lhs.nr = p.xr; center_lhs.nc = p.xr;
	matrix center_rhs;
	center_rhs.nr = p.xr; center_rhs.nc = p.xc;

	preproc_linear_data pp;
	if(local_rank == 0) {
		center_lhs.buf = malloc(ATA_size*sizeof(fp_type));
		center_rhs.buf = malloc(ATB_size*sizeof(fp_type));
		memset(center_lhs.buf, 0, ATA_size*sizeof(fp_type));
		memset(center_rhs.buf, 0, ATB_size*sizeof(fp_type));
		if(!p.A.buf) {
			center_ATA.buf = malloc(ATA_size*sizeof(fp_type));
			center_ATB.buf = malloc(ATB_size*sizeof(fp_type));
			memset(center_ATA.buf, 0, ATA_size*sizeof(fp_type));
			memset(center_ATB.buf, 0, ATB_size*sizeof(fp_type));

		}
		else {
			preprocess_worker_data(p, &pp);
			center_ATA = pp.lhs;
			center_ATB = pp.rhs;
		}
	}
	else if(local_rank > 0) {
		preprocess_worker_data(p, &pp);
	}
	
	//Reduce SUM(A^TA) into the center
	if(local_rank > 0)
		MPI_Reduce(pp.lhs.buf, NULL, ATA_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);
	else if(local_rank == 0)
		MPI_Reduce(center_ATA.buf, center_lhs.buf, ATA_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);
	/*We synchronize after reducing LHS, before reducing RHS*/
	MPI_Barrier(ls_comm);
	
	//Reduce SUM(A^TB) into the center
	if(local_rank > 0)
		MPI_Reduce(pp.rhs.buf, NULL, ATB_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);
	else if(local_rank == 0)
		MPI_Reduce(center_ATB.buf, center_rhs.buf, ATB_size, MPI_FP_TYPE, MPI_SUM, 0, ls_comm);
	MPI_Barrier(ls_comm);

	if(local_rank == 0) {
		/* Set the preprocessed data at the center to
		 * the sums and clean up the center's local vars
		 * Note the solution will be stored in pp.rhs
		 */
		pp.lhs = center_lhs;
		pp.rhs = center_rhs;
		free(center_ATA.buf);
		free(center_ATB.buf);

		if(p.reg_type == L2)
			l2_ols_solution(p, pp, sol);
		else if(p.reg_type == L1)
			l1_ols_admm(p, pp, ahp, sol);
	}

	//Clean heaps	
	if(local_rank == 0) {
		free(center_lhs.buf);
		free(center_rhs.buf);
	}
	if(local_rank > 0) {
		free(pp.rhs.buf);
		free(pp.lhs.buf);
	}
	
	return(1);
}
