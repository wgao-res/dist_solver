#include "linalg_core.h"
#include <omp.h>

int copy_matrix(matrix A, matrix *A_new)
{
	A_new->nr = A.nr;
	A_new->nc = A.nc;
	A_new->buf = malloc(A.nr*A.nc*sizeof(fp_type));
	if (A_new->buf) {
		memcpy(A_new->buf, A.buf, A.nr*A.nc*sizeof(fp_type));
		return(1);
	}
	else
		return(-1); 
}

void add_identity(matrix A, fp_type beta)
{
	for(int i = 0; i < A.nr; i++)
		A.buf[i*A.nr + i] += beta;
}	

void add_scaled_matrix(fp_type alpha, matrix A, fp_type beta, matrix B)
{
	//BLAS parameters
	int n = B.nr*B.nc;
	int incx = 1;
	SCAL_(&n, &beta, B.buf, &incx);

	int incy = 1;
	if(A.buf && alpha != 0)
		AXPY_(&n, &alpha, A.buf, &incy, B.buf, &incx);
}

//For when we want to add A, but the transpose of A is stored
void add_transpose_matrix(fp_type alpha, matrix A, fp_type beta, matrix B)
{
	if(A.buf) {
		for(int j = 0; j < B.nc; j++)
			for(int i = 0; i < B.nr; i++)
				B.buf[i+j*B.nr] = beta*B.buf[i+j*B.nr] + alpha*A.buf[j+i*A.nr];
	}
	else
		add_scaled_matrix(alpha, A, beta, B);
}

void print_matrix(matrix A, int cols, char *heading) {
	if(cols == 0 || cols > A.nc)
		cols = A.nc;
	printf("%s\n", heading);
	for(int i = 0; i < A.nr; i++) {
		for(int j = 0; j < cols; j++) {
			printf("%.10f", A.buf[j*A.nr+i]); //Column order
			if(j < A.nc - 1)
				printf(", ");
			else
				printf("\n");
		}
	}
}

fp_type max_entry(matrix A) {
	fp_type max_seen;
	for(int i = 0; i < A.nr*A.nc; i++) {
		if(i == 0)
			max_seen = A.buf[i];
		else
			if(A.buf[i] > max_seen)
				max_seen = A.buf[i];
	}
	return(max_seen);
}

fp_type frob_norm(fp_type alpha, matrix A, fp_type beta, matrix B) {
	fp_type running_sum = 0;
	if(B.buf && beta != 0) {
		for(int i = 0; i < A.nr*A.nc; i++)
			running_sum += pow(alpha*A.buf[i] + beta*B.buf[i], 2);
	}
	else
		for(int i = 0; i < A.nr*A.nc; i++)
			running_sum += pow(alpha*A.buf[i], 2);
	return(sqrt(running_sum));
}

fp_type dist_l1_subgrad(matrix D, matrix Y, matrix Z, fp_type scale, int is_Zt) {
	fp_type running_sum = 0;
	for(int i = 0; i < D.nc; i++)
		for(int j = 0; j < D.nr; j++) {
			int D_idx = i*D.nr + j;
			int Z_idx;
			if(!is_Zt)
				Z_idx = D_idx;
			else
				Z_idx = j*Z.nr + i;
			fp_type dual_var = D.buf[D_idx];
			fp_type center;
			if(Z.buf)
				center = Z.buf[Z_idx];
			else
				center = 0;
			if(Y.buf[D_idx] - center > 0)
				running_sum += pow(dual_var - scale, 2);
			else if(Y.buf[D_idx] - center < 0)
				running_sum += pow(dual_var + scale, 2);
			else {
				if(dual_var < -scale)
					running_sum += pow(dual_var + scale, 2); 
				else if(dual_var > scale)
					running_sum += pow(dual_var - scale, 2); 
			}
		}
	return(sqrt(running_sum));
}

void apply_pointwise(matrix A, matrix fA, univar_f func) {
	#pragma omp parallel
	{
		int omp_trank = omp_get_thread_num();
		int omp_tnum = omp_get_num_threads();

		int idx = omp_trank;

		while(idx < A.nr*A.nc) { 
			fA.buf[idx] = (*func)(A.buf[idx]);
			idx += omp_tnum;
		}
	}
}

fp_type soft_threshold(fp_type x) {
	if(x > 1)
		return(x - 1);
	else if(x < -1)
		return(x + 1);
	else
		return(0);
}

void pointwise_soft_thresholding(matrix X, matrix Z, matrix fX, fp_type inc, int is_Zt) {
	#pragma omp parallel
	{
		int omp_trank = omp_get_thread_num();
		int omp_tnum = omp_get_num_threads();
		int idx = omp_trank;

		while(idx < X.nr*X.nc) {
			int Z_idx;
			if(!is_Zt)
				Z_idx = idx;
			else {
				int row = idx % X.nc;
				int col = (idx - row) / X.nc;
				Z_idx = row*Z.nr + col;
			}
			fp_type center;
			if(Z.buf)
				center = Z.buf[Z_idx];
			else
				center = 0;

			if(X.buf[idx] > center + inc)
				fX.buf[idx] = X.buf[idx] - inc;
			else if(X.buf[idx] < center - inc)
				fX.buf[idx] = X.buf[idx] + inc;
			else
				fX.buf[idx] = center;
			idx += omp_tnum;
		}
	}
}
