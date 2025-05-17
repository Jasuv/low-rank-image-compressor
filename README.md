# Low-rank Image Compressor
**Singular Value Decomposition (SVD)** is an important tool in matrix factorization. 
```
A=UΣVᵀ
```
SVD is a method to decompose matrix A *m×n* into 3 simpler matricies:
- **U**: An *m×m* orthogonal matrix whose columns are *left singular vectors* of A
- **Σ**: A diagonal *m×n* matrix of the *singular values* of A in descending order
- **Vᵀ**: A transposed *n×n* orthogonal matrix whose columns are *right singular vectors* of A

with a small modification to the SVD equation, we can get the best rank *k* approximation to A.
```
Aₖ=∑ᵢ₌₁ᵏσᵢuᵢvᵢᵀ
```
- **U**: An ***m×k*** orthogonal matrix whose columns are *left singular vectors* of **Aₖ**
- **Σ**: A diagonal ***kxk*** matrix of the *singular values* of **Aₖ** in descending order
- **Vᵀ**: A transposed ***k×n*** orthogonal matrix whose columns are *right singular vectors* of **Aₖ**
### How to Build/Run
#### SVD test script (modify A, m, n, k in code)
```
gcc svd_test.c matrix_functions.c -lm -o svd_test
./svd_test matrix
```
#### Image compressor script
```
gcc image_compressor.c matrix_functions.c -lpng -lm -o image_compressor
(example) ./image_compressor mona_lisa.png mona_lisa_compressed.png 50
```
#### Usage:
```
./image_compressor <input>.png <output-name>.png <k-rank>
```
