/**************************************************************
* Filename:: 	svd_test.c
* Author:: 		Eric Ahsue
* Github::  	Jasuv
* Description:: tests SVD code with example output
*
**************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "matrix_functions.h"

// prints matricies in row-major
void print_matrix(const char *name, double **A, int p, int q) {
	printf("\n%s\n", name);
	for (int i = 0; i < p; i++) {
		printf("  [");
		for (int j = 0; j < q; j++) {
			printf("%6.2f ", A[i][j]);
		}
		printf("]\n");
	}
	return;
}

// low-rank approximation
// Aₖ=∑ᵢ₌₁ᵏσᵢuᵢvᵢᵀ
double **low_rank_approximation(double **A, int p, int q, int k) {
	double **At = transpose(A, p, q);
	double **AtA = multiply(At, A, q, p, q);
	double **S = alloc_matrix(k, k);
	double **V = alloc_matrix(q, k);

	// solve for V
	// AᵀA=VΣᵀΣVᵀ
	power_iteration(AtA, V, S, q, k);

	// transpose V
	double **Vt = transpose(V, q, k);

	// solve for U
	// U=AVΣ⁻¹
	double **AV = multiply(A, V, p, q, k);
	double **S_inv = diagonal_inverse(S, k);
	double **U = multiply(AV, S_inv, p, k, k);

	// calculate compressed A
 	// Aₖ=UΣVᵀ
	double **US = multiply(U, S, p, k, k); 
	double **Ak = multiply(US, Vt, p, k, q);

	// print matricies
	print_matrix("At", At, q, p);
	print_matrix("AtA", AtA, q, q);
	print_matrix("U", U, p, k);
	print_matrix("S", S, k, k);
	print_matrix("Vt", Vt, k, q);

	// cleanup
	free_matrix(At, q);
	free_matrix(AtA, q);
	free_matrix(S, k);
	free_matrix(V, q);
	free_matrix(Vt, k);
	free_matrix(AV, p);
	free_matrix(S_inv, k);
	free_matrix(U, p);
	free_matrix(US, p);

	return Ak;
}

int main(int argc, char *argv[]) {
	// set random seed for power iteration random vector
	srand(time(NULL));

	// test vars
	int p = 3;
	int q = 3;
	int k = 3; // k=r

	printf("p = %d, q = %d, k = %d\n", p, q, k);

	// test matrix
	double **A = alloc_matrix(p, q);
	A[0][0] = 3;
	A[0][1] = 1;
	A[0][2] = 2;
	A[1][0] = 2;
	A[1][1] = 3;
	A[1][2] = 1;
	A[2][0] = 1;
	A[2][1] = 2;
	A[2][2] = 3;

	print_matrix("A", A, p, q);

	// compress A
	double **Ak = low_rank_approximation(A, p, q, k);

	print_matrix("Ak", Ak, p, q);

	// cleanup
	free_matrix(A, p);
	free_matrix(Ak, p);

	return 0;
}
