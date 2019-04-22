#include "linalg_core.h"

struct ADMM_params {
	fp_type rho;
	int max_iter;
	fp_type primal_stop;
	fp_type dual_stop;
	int termination_test_freq;
};
typedef struct ADMM_params ADMM_params;

/* DISTRIBUTED LEAST SQUARES PROBLEMS */

/* A struct for linear problems of the form
 * \min_X 1/2 \|AX - B\|_2^2 - <C,X> + reg_lambda h(X - Z)
 * where A, B, C are local data.
 *
 */
struct linear_data {
	//dimensions of X
	int xr;
	int xc;

	//regularization parameter
	enum regularizer_type reg_type;
	fp_type reg_lambda;
	
	//problem data
	matrix A;
	matrix B;
	matrix Z;
	matrix C;
	char transa;
	char transb;
	char transz;
	char transc;

};
typedef struct linear_data linear_data;

struct preproc_linear_data {
	matrix lhs;
	matrix rhs;
};
typedef struct preproc_linear_data preproc_linear_data;

int preprocess_worker_data(linear_data p, preproc_linear_data *pp);
int admm_rho_cholesky(matrix ATA, fp_type rho);
int l2_ols_solution(linear_data p, preproc_linear_data pp, matrix *sol);
int l1_ols_admm(linear_data p, preproc_linear_data pp, ADMM_params ahp, matrix *sol);

int jt_ls_solver(MPI_Comm ls_comm, linear_data p, ADMM_params ahp, matrix *sol);
int consensus_admm_ls(MPI_Comm ls_comm, linear_data p, ADMM_params ahp, matrix *sol);

typedef int (*solver_signature)(MPI_Comm, linear_data, ADMM_params, matrix *);
/*******************/

