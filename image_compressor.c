/**************************************************************
 * Filename::    image_compressor.c
 * Author::      Eric Ahsue
 * Github::      Jasuv
 * Description:: uses low-rank approximation to compres PNGs
 *
 **************************************************************/

#include <math.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix_functions.h"

// low-rank approximation
// Aₖ=∑ᵢ₌₁ᵏσᵢuᵢvᵢᵀ
double **low_rank_approximation(double **A, int p, int q, int k) {
	double **At = transpose(A, p, q);
	double **AtA = multiply(At, A, q, p, q); // to solve V
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

// convert PNG to matrix
double **png_to_matrix(const char *filename, int *w, int *h) {
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		printf("cannot find file %s\n", filename);
		return NULL;
	}

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop info = png_create_info_struct(png);
	if (setjmp(png_jmpbuf(png)))
		return NULL;

	png_init_io(png, fp);
	png_read_info(png, info);

	*w = png_get_image_width(png, info);
	*h = png_get_image_height(png, info);

	png_bytep *rows = malloc(sizeof(png_bytep) * (*h));
	unsigned char *raw = malloc((*w) * (*h));
	for (int y = 0; y < *h; y++) {
		rows[y] = raw + y * (*w);
	}
	png_read_image(png, rows);
	fclose(fp);

	double **mat = alloc_matrix(*h, *w);
	for (int i = 0; i < *h; i++) {
		for (int j = 0; j < *w; j++) {
			mat[i][j] = (double)raw[i * (*w) + j];
		}
	}

	free(raw);
	free(rows);
	png_destroy_read_struct(&png, &info, NULL);
	return mat;
}

// save matrix as grayscale png
void matrix_to_png(const char *filename, double **mat, int w, int h) {
	FILE *fp = fopen(filename, "wb");

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop info = png_create_info_struct(png);
	if (setjmp(png_jmpbuf(png))) return;

	png_init_io(png, fp);
	png_set_IHDR(png, info, w, h, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
				 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_write_info(png, info);

	png_bytep *rows = malloc(sizeof(png_bytep) * h);
	unsigned char *data = malloc(w * h);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			double val = round(mat[i][j]);
			data[i * w + j] = (val < 0) ? 0 : (val > 255 ? 255 : val);
		}
	}

	for (int y = 0; y < h; y++) {
		rows[y] = data + y * w;
	}
	png_write_image(png, rows);
	png_write_end(png, NULL);
	fclose(fp);
	free(rows);
	free(data);
	png_destroy_write_struct(&png, &info);
}

int main(int argc, char *argv[]) {
	// set random seed for power iteration
	srand(time(NULL));

	if (argc != 4) {
		printf("usage: %s input.png output.png k-rank\n", argv[0]);
		return 1;
	}

	int w = 0;
	int h = 0;
	int k = atoi(argv[3]);

	// convert grayscale png to pixel matrix
	double **A = png_to_matrix(argv[1], &w, &h);

	// k must be < or == r (rank of A)
	if (k > w || k > h) {
		printf("k too large for image size.\n");
		return 1;
	}

	// calculate compressed pixel matrix
	double **Ak = low_rank_approximation(A, h, w, k);

	// turn pixel matrix back into png
	// should change later to my own custom image format
	matrix_to_png(argv[2], Ak, w, h);
	printf("compressed %s to %s with k=%d singular values\n", argv[1], argv[2], k);

	free_matrix(A, h);
	free_matrix(Ak, h);

	return 0;
}
