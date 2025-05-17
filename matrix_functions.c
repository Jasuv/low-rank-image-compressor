/**************************************************************
* Filename::    matrix_functions.c
* Author::      Eric Ahsue
* Github::      Jasuv
* Description:: contains matrix functions
*
**************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITER 1000
#define EPSILON 1e-6

// allocate an empty (m, n) matrix
double **alloc_matrix(int m, int n) {
    double **A = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        A[i] = malloc(n * sizeof(double));
        memset(A[i], 0, n * sizeof(double));
    }
    return A;
}

// free all cols and rows of matrix
void free_matrix(double **A, int m) {
    for (int i = 0; i < m; i++) {
        free(A[i]);
    }
    free(A);
    return;
}

// row-column rule matrix multiplication
// A(m, n) B(n, r) AB(m, r)
double **multiply(double **A, double **B, int m, int n, int r) {
    double **AB = alloc_matrix(m, r);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < r; j++) {
            for (int k = 0; k < n; k++) {
                AB[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return AB;
}

// matrix-vector multiplication
// A(m, n) v(n, 1) Av(m, 1)
double *multiply_vec(double **A, double *v, int m, int n) {
    double *Av = malloc(m * sizeof(double));
    memset(Av, 0, m * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            Av[i] += A[i][j] * v[j];
        }
    }
    return Av;
}

// inverse of diagonal matrix
// a=1/a
double **diagonal_inverse(double **A, int n) {
	double **A_inv = alloc_matrix(n, n);
	for (int i = 0; i < n; i++) {
		A_inv[i][i] = 1 / A[i][i];
	}
	return A_inv;
}

// transposes matrix
double **transpose(double **A, int m, int n) {
    double **At = alloc_matrix(n, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            At[j][i] = A[i][j];
        }
    }
    return At;
}

// dot product
double dot(double *a, double *b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
	}
    return sum;
}

// normalize vector
void normalize(double *v, int n) {
    double norm = sqrt(dot(v, v, n));
    for (int i = 0; i < n; i++) {
        v[i] /= norm;
	}
	return;
}

// power iteration method to find top k eigen vectors
void power_iteration(double **A, double **eigvecs, double **S, int n, int k) {
    for (int i = 0; i < k; i++) {
        // initialize random vector
        double *v = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            v[j] = rand() / (double)RAND_MAX;
        }

        // perform power iterative loop
        for (int iter = 0; iter < MAX_ITER; iter++) {

			// calcualte Av (n, 1) matrix
			double *Av = multiply_vec(A, v, n, n);

			// deflation
			// orthogonalize Av with previous eigen vectors
            for (int y = 0; y < i; y++) {
                // get previous eigen vector
				double proj = 0;
				for (int x = 0; x < n; x++) {
					proj += Av[x] * eigvecs[x][y];
				}

                // deflate Av
				for (int x = 0; x < n; x++) {
					Av[x] -= proj * eigvecs[x][y];
				}
            }

            // normalize Av
            normalize(Av, n);

            // checks if approximation is good enough
            double diff = 0;
            for (int j = 0; j < n; j++) {
                diff += fabs(v[j] - Av[j]);
            }
            if (diff < EPSILON) {
            	memcpy(v, Av, n * sizeof(double));
				free(Av);
				break;
			}

            // update v as Av
            memcpy(v, Av, n * sizeof(double));
            free(Av);
        }

        // copy the eigen vector to eigvecs (in row-major layout)
        for (int j = 0; j < n; j++) {
            eigvecs[j][i] = v[j];
        }

        // calculate and store singular values into Σ
        double *Av = multiply_vec(A, v, n, n);
        S[i][i] = sqrt(dot(v, Av, n));

        // clean up
        free(v);
        free(Av);
    }
    
    return;
}
