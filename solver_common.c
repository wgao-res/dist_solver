#include "solver_core.h"

int preprocess_worker_data(linear_data p, preproc_linear_data *pp)
{
	/* Preprocess the data at each worker to get
	 * RHS <-- A^TB + C
	 * LHS <-- L where L lower triangular block of A^TA
	 *
	 * NOTE: we allocate new storage for LHS, RHS
	 * Original buffers in p are unaffected
	 */

	matrix rhs;
	if(p.transa == 'N')
		rhs.nr = p.A.nc;
	else if(p.transa == 'T')
		rhs.nr = p.A.nr;
	if(p.transb == 'N')
		rhs.nc = p.B.nc;
	else if(p.transb == 'T')
		rhs.nc = p.B.nr;
	
	int rhs_size = rhs.nr*rhs.nc;
	rhs.buf = malloc(rhs_size*sizeof(fp_type));
	if(!rhs.buf)
		return(-1);
	else
		memset(rhs.buf, 0, rhs_size*sizeof(fp_type));
	
	/* GEMM params for RHS <-- A^TB + C.
	 * If no C term, we leave the zero matrix
	 * In the GEMM_ notation,
	 * A <-- A^T
	 * B <-- B
	 */
	if(p.transc == 'N' && p.C.buf)
		add_scaled_matrix(1, p.C, 0, rhs);
	else if(p.transc == 'T' && p.C.buf)
		add_transpose_matrix(1, p.C, 0, rhs);

	char transa;
	char transb;

       	if(p.transa == 'N')	
		transa = 'T';
	else if(p.transa == 'T')
		transa = 'N';
	if(p.transb == 'N')
		transb = 'N';
	else if(p.transb == 'T')
		transb = 'T';

	int M = rhs.nr; //rows of A^TB
	int N = rhs.nc; //columns of A^TB
	int K;	//internal dim (between ops in product)
       	if(p.transb == 'N')
		K = p.B.nr;
       	else if(p.transb == 'T')
		K = p.B.nc;
	fp_type alpha = 1;
	int lda = p.A.nr; //leading dimension of A
	int ldb = p.B.nr; //leading dimension of B
	fp_type beta = 0;
	int ldc = rhs.nr;

	//Call GEMM to set RHS to A^TB + C
	GEMM_(&transa, &transb, &M, &N, &K, &alpha, p.A.buf, &lda, p.B.buf, &ldb, &beta, rhs.buf, &ldc);
	
	//Compute LHS. Initialize zero matrix for it
	//LHS should have the same dimensions as A^TA
	matrix lhs;
	if(p.transa == 'N') {
		lhs.nr = p.A.nc;
		lhs.nc = p.A.nc;
	}
	else if(p.transa == 'T') {
		lhs.nr = p.A.nr;
		lhs.nc = p.A.nr;
	}
	int lhs_size = lhs.nr*lhs.nc;
      	lhs.buf = malloc(lhs_size*sizeof(fp_type));
	if(!lhs.buf)
		return(-1);
	else
		memset(lhs.buf, 0, lhs_size*sizeof(fp_type));

	/* SYRK params for LHS <-- A^TA
	 * Note that SYRK already stores only the lower triangle
	 * when given uplo = 'L' (i.e will have zeros in upper triangle)
	 * Thus we need to use the same UPLO for SYRK and POTRF
	 */
	char uplo = 'L';
	if(p.transa == 'N')
		transa = 'T';
	else if(p.transa == 'T')
		transa = 'N';
	N = lhs.nr;
	if(p.transa == 'N')
		K = p.A.nr;
	else if(p.transa == 'T')
		K = p.A.nc;
	alpha = 1;
	lda = p.A.nr; 
	beta = 0;
	ldc = lhs.nr;

	//Call SYRK_ to set LHS to A^TA
	SYRK_(&uplo, &transa, &N, &K, &alpha, p.A.buf, &lda, &beta, lhs.buf, &ldc);
	pp->lhs = lhs;
	pp->rhs = rhs;
	return(1);
}


int admm_rho_cholesky(matrix ATA, fp_type rho) {
	/* It is assumed that ATA stores a PD matrix
	 * A^TA as the lower triangle. This function
	 * destructively modifeis ATA to the
	 * lower Cholesky factorization L
	 * LL^T = A^TA + \rho*I
	 */
	if(rho > 0)
		add_identity(ATA, rho);

	/* POTRF params for L, LL^T = A^TA + rho*I
	 */
	char uplo = 'L';
	int N = ATA.nr;
	int lda = ATA.nr;
	int info;

	//Call POTRF
	POTRF_(&uplo, &N, ATA.buf, &lda, &info);
	
	return(1);
}


int l2_ols_solution(linear_data p, preproc_linear_data pp, matrix *sol){
	matrix reg_lhs;
	copy_matrix(pp.lhs, &reg_lhs);
	if(p.reg_type == L2 && p.reg_lambda > 0) {
		add_identity(reg_lhs, p.reg_lambda);
		if(p.Z.buf && p.transz == 'N')
			add_scaled_matrix(p.reg_lambda, p.Z, 1, reg_lhs);
		else if(p.Z.buf && p.transz == 'T')
			add_transpose_matrix(p.reg_lambda, p.Z, 1, reg_lhs);
	}
	//POSV params 
	char uplo = 'L';	
	int N = p.xr;
	int nrhs = p.xc;
	int lda = p.xr;
	int ldb = p.xr;
	int info;
	
	copy_matrix(pp.rhs, sol);
	POSV_(&uplo, &N, &nrhs, reg_lhs.buf, &lda, sol->buf, &ldb, &info);

	free(reg_lhs.buf);
	return(1);
}

int l1_ols_admm(linear_data p, preproc_linear_data pp, ADMM_params ahp, matrix *sol) {
	
	/* We use ADMM to solve the L1-regularized OLS problem
	 * min_{X,Y} 1/2*\|AX - B\|^2 - <C,X> + \lambda/2\|Y - Z\|_1
	 * 		s.t X - Y = 0
	 * The steps are as follows:
	 * 1 update X. Store pp.rhs + \rho*Y - Dual
	 *   in the variable X and then solve the regularized
	 *   quadratic.
	 * 2 update Y. the FOC is
	 *   \rho*Y = \rho*X + Dual - \lambda/2 \partial\|Y-Z\|_1
	 *   so the solution is given by:
	 *   	Y = X' - inc	if X' > Z + inc
	 *   	  = X' + inc	if X' < Z - inc
	 *   	  = Z		else
	 *   	where X' = X + 1/\rho*Dual, inc = \lambda/(2\rho)
	 * 	
	 * 3 update Dual. Dual <- Dual + \rho(X - Y)  
	 */

	/* ADMM parameters and tests */
	fp_type rho = ahp.rho;

	int iter = 0;
	int primal_test = FALSE;
	int dual_test = FALSE;
	/*******************/

	/* ADMM auxiliary variables */
	matrix X;
	matrix Y;
	matrix Dual;
	matrix admm_lhs;
	/*******************/

	if(copy_matrix(pp.rhs, &X) == -1
		|| copy_matrix(pp.rhs, &Y) == -1
		|| copy_matrix(pp.rhs, &Dual) == -1
		|| copy_matrix(pp.lhs, &admm_lhs) == -1)
			return(-1);

	memset(Dual.buf, 0, X.nr*X.nc*sizeof(fp_type));
	add_identity(admm_lhs, rho);

	int is_Zt = FALSE;
	if(p.transz == 'T')
		is_Zt = TRUE;

	while( !(primal_test && dual_test) && iter < ahp.max_iter) {
		// X update
		/* set X to A^TB + C = pp.rhs
		 * then add \rho Y - Dual
		 */
		memcpy(X.buf, pp.rhs.buf, X.nr*X.nc*sizeof(fp_type));

		add_scaled_matrix(rho, Y, 1, X);
		add_scaled_matrix(-1, Dual, 1, X);
		
		/* Note that the first call to POSV_ overwrites
		 * admm_lhs with the lower triangle of the Cholesky
		 * factorization. Thus we call TRTRS_ after the
		 * first iteration to do triangular solve.
		 */
		if(iter == 0) {
			//POSV params 
			char uplo = 'L';	
			int N = p.xr;
			int nrhs = p.xc;
			int lda = p.xr;
			int ldb = p.xr;
			int info;
			POSV_(&uplo, &N, &nrhs, admm_lhs.buf, &lda, X.buf, &ldb, &info);
		}
		else {
			//TRTRS params
			char uplo = 'L';
			char trans = 'N';
			char diag = 'N';
			int N = p.xr;
			int nrhs = p.xc;
			int lda = p.xr;
			int ldb = p.xr;
			int info;
			TRTRS_(&uplo, &trans, &diag, &N, &nrhs, admm_lhs.buf, &lda, X.buf, &ldb, &info);

			trans = 'T';
			TRTRS_(&uplo, &trans, &diag, &N, &nrhs, admm_lhs.buf, &lda, X.buf, &ldb, &info);
		}
		
		//Y update
		memcpy(Y.buf, X.buf, X.nr*X.nc*sizeof(fp_type));
		add_scaled_matrix(1/rho, Dual, 1, Y); 
		pointwise_soft_thresholding(Y, p.Z, Y, p.reg_lambda/(2*rho), is_Zt);
		
		//Dual update
		add_scaled_matrix(rho, X, 1, Dual);
		add_scaled_matrix(-1*rho, Y, 1, Dual);

		//Test for termination every 5 ADMM steps
		if(iter % ahp.termination_test_freq  == 0) {

			/*A common primal/dual termination
			 * criterion for ADMM.
			 * primal gap: X-Y
			 * dual gap:
			 * 	-W + \partial \|Y-Z\|_1
			 * 	AT(AX-B) + W
			 */
			//dual
			fp_type primal_gap = frob_norm(1, X, -1, Y);
			if(primal_gap < ahp.primal_stop)
				primal_test = TRUE;
			else
				primal_test = FALSE;
				
			//primal
			fp_type dual_gap = dist_l1_subgrad(Dual, Y, p.Z, p.reg_lambda/2, is_Zt);
			matrix X_grad_diff;
			copy_matrix(Dual, &X_grad_diff);
			add_scaled_matrix(-1, pp.rhs, 1, X_grad_diff);

			//SYMM_
			char side = 'L';
			char uplo = 'L';
			int M = X.nr;
			int N = X.nc;
			fp_type alpha = 1;
			int lda = admm_lhs.nr;
			int ldb = X.nr;
			fp_type beta = 1;
			int ldc = X.nr;
			
			SYMM_(&side, &uplo, &M, &N, &alpha, pp.lhs.buf, &lda, X.buf, &ldb, &beta, X_grad_diff.buf, &ldc);
			
			dual_gap += frob_norm(1, X_grad_diff, 0, X_grad_diff);
			if(dual_gap < ahp.dual_stop)
				dual_test = TRUE;
			else
				dual_test = FALSE;
			
			free(X_grad_diff.buf);
			#ifdef _VERBOSE_
			printf("iter: %d dual: %f primal: %f\n", iter, dual_gap, primal_gap);
			#endif
		}	
		iter += 1;
	}

	copy_matrix(Y, sol);

	//cleanup ADMM auxiliary variables
	free(X.buf);
	free(Y.buf);
	free(Dual.buf);
	free(admm_lhs.buf);
	return(1);
}
