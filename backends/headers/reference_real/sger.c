/**
 * @file sger.c
 * @brief Reference implementation of SGER (Single precision outer product)
 *
 * SGER: A := alpha*x*y^T + A
 *
 * Rank-1 update of a general matrix
 */

#include <blas_reference.h>

/**
 * Single precision outer product (rank-1 update)
 *
 * @param m Number of rows in A
 * @param n Number of columns in A
 * @param alpha Scalar
 * @param x Vector x (m elements)
 * @param incx Stride for x
 * @param y Vector y (n elements)
 * @param incy Stride for y
 * @param A Matrix A (m x n, column-major)
 * @param lda Leading dimension of A
 */
void sger_ref(int m, int n, float alpha, const float *x, int incx, const float *y, int incy,
              float *A, int lda) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0f) return;
    
    int jy = 0;
    for (int j = 0; j < n; j++) {
        float temp = alpha * y[jy];
        int ix = 0;
        for (int i = 0; i < m; i++) {
            A[i + j * lda] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}
