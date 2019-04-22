#ifndef _STANDARD_HEADERS_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>
#define _STANDARD_HEADERS_
#endif

#define _DOUBLE_PRECISION_

#ifdef _DOUBLE_PRECISION_
typedef double fp_type;
#define MPI_FP_TYPE MPI_DOUBLE
#else
typedef float fp_type;
#define MPI_FP_TYPE MPI_FLOAT
#endif

#ifdef _DOUBLE_PRECISION_
extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
extern void dgesv_(int *n, int *nrhs,  double *a,  int *lda,  
           int *ipivot, double *b, int *ldb, int *info);
extern void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda, double *b, int *ldb, int *info);
extern void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);
extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
extern void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
extern void dsymm_(char *side, char *uplo, int *M, int *N, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);

#else
extern void sscal_(int *n, float *alpha, float *x, int *incx);
extern void saxpy_(int *n, float *alpha, float *x, int *incx, float *y, int *incy);
extern void sgesv_(int *n, int *nrhs,  float *a,  int *lda,  
           int *ipivot, float *b, int *ldb, int *info);
extern void sposv_(char *uplo, int *n, int *nrhs, float *A, int *lda, float *b, int *ldb, int *info);
extern void strtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, float *A, int *lda, float *B, int *ldb, int *info);
extern void spotrf_(char *uplo, int *n, float *A, int *lda, int *info);
extern void strmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *A, int *lda, float *B, int *ldb);
extern void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
extern void ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha, float *A, int *lda, float *beta, float *C, int *ldc);
extern void ssymm_(char *side, char *uplo, int *M, int *N, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
#endif

#ifdef _DOUBLE_PRECISION_
#define SCAL_ dscal_
#define AXPY_ daxpy_
#define GESV_ dgesv_
#define POSV_ dposv_
#define TRTRS_ dtrtrs_
#define POTRF_ dpotrf_
#define TRMM_ dtrmm_
#define GEMM_ dgemm_
#define SYRK_ dsyrk_
#define SYMM_ dsymm_
#else
#define SCAL_ sscal_
#define AXPY_ saxpy_
#define GESV_ sgesv_
#define POSV_ sposv_
#define TRTRS_ strtrs_
#define POTRF_ spotrf_
#define TRMM_ strmm_
#define GEMM_ sgemm_
#define SYRK_ ssyrk_
#define SYMM_ ssymm_
#endif

#ifndef _COMMON_DEFS_
#define TRUE 1
#define FALSE 0

/* Matrix structs */
struct matrix {
	fp_type *buf;
	int nr;
	int nc;
};
typedef struct matrix matrix;

struct int_matrix {
	int *buf;
	int nr;
	int nc;
};
typedef struct int_matrix int_matrix;
/*******************/

/* Matrix operations */
int copy_matrix(matrix A, matrix *A_new);
/* Allocates buffer for A_new and copy A.buf 
 * return: 1 if successful, -1 if malloc fails */

void add_identity(matrix A, fp_type beta);
// Adds \beta*I to A

void add_scaled_matrix(fp_type alpha, matrix A, fp_type beta, matrix B);
//  Stores \alpha*A + \beta*B into B

void add_transpose_matrix(fp_type alpha, matrix A, fp_type beta, matrix B);
// Stores \alpha*A^T + \beta*B into B

void print_matrix(matrix A, int cols, char *heading);
fp_type max_entry(matrix A);
fp_type frob_norm(fp_type alpha, matrix A, fp_type beta, matrix B);
//The Frobenius norm of \alpha*A + \beta*B
fp_type dist_l1_subgrad(matrix D, matrix Y, matrix Z, fp_type scale, int is_Zt);
/*******************/

/* Pointwise function applications */
typedef fp_type (univar_f)(fp_type);
void apply_pointwise(matrix A, matrix fA, univar_f func);
// Writes f(A) pointwise into fA
fp_type soft_threshold(fp_type x);
void pointwise_soft_thresholding(matrix X, matrix Z, matrix fX, fp_type inc, int is_Zt);
/*******************/

/* Linear regression specification */ 
enum regularizer_type {L1, L2};

#define _COMMON_DEFS_
#endif
