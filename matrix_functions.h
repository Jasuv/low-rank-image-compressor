/**************************************************************
* Filename::    matrix_functions.h
* Author::      Eric Ahsue
* Github::      Jasuv
* Description:: contains matrix function prototypes
*
**************************************************************/

#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H

// matrix functions
double **alloc_matrix(int p, int q);
void free_matrix(double **A, int p);
double **multiply(double **A, double **B, int p, int q, int r);
double *multiply_vec(double **A, double **B, int p, int q);
double **diagonal_inverse(double **A, int n);
double **transpose(double **A, int p, int q);
double dot(double *a, double *b, int n);
void normalize(double *v, int n);

/* power iteration method to find top k eigen vectors
 * 
 * params: matrix A, eigen vector arr (U||V), singular values matrix, square dimension, desired rank
 * 
 * for desired rank (k)
 * 		create a random vector v(n, 1)
 * 		for MAX_ITERATIONS
 * 			multiply A and v -> Av
 * 			deflation (orthogonalize Av with previous eigen vector)
 *			normalize Av -> converge to top eigen vector
 *			check if eigen vector approximation (Av)
 * 		add Av to to eigen vectors
 * 		add singular value to S
 * 
 */
void power_iteration(double **A, double **eigvecs, double **S, int n, int k);

#endif
