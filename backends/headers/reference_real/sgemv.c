/**
 * @file sgemv.c
 * @brief Reference implementation of SGEMV (Single precision general matrix-vector multiply)
 *
 * SGEMV: y := alpha*A*x + beta*y (or A^T*x or A^H*x)
 *
 * y[i] = alpha * sum_j(A[i,j] * x[j]) + beta * y[i]
 * or with transpose:
 * y[j] = alpha * sum_i(A[i,j] * x[i]) + beta * y[j]
 */

#include <blas_reference.h>
#include <string.h>

/**
 * Single precision general matrix-vector multiply
 *
 * @param trans Transpose: 'N', 'T', 'C' (no, transpose, conjugate transpose)
 * @param m Number of rows in A
 * @param n Number of columns in A
 * @param alpha Scalar multiplier for A*x
 * @param A Matrix A (m x n, stored column-major)
 * @param lda Leading dimension of A (>= m)
 * @param x Input vector (n elements if trans='N', m if trans='T'/'C')
 * @param incx Stride for x
 * @param beta Scalar multiplier for y
 * @param y Output vector (m elements if trans='N', n if trans='T'/'C')
 * @param incy Stride for y
 */
void sgemv_ref(char trans, int m, int n, float alpha, const float *A, int lda,
               const float *x, int incx, float beta, float *y, int incy) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0f && beta == 1.0f) return;
    
    if (trans == 'N' || trans == 'n') {
        /* y := alpha*A*x + beta*y */
        
        /* Scale y by beta */
        if (beta == 0.0f) {
            int iy = 0;
            for (int i = 0; i < m; i++) {
                y[iy] = 0.0f;
                iy += incy;
            }
        } else if (beta != 1.0f) {
            int iy = 0;
            for (int i = 0; i < m; i++) {
                y[iy] *= beta;
                iy += incy;
            }
        }
        
        if (alpha == 0.0f) return;
        
        /* Compute alpha*A*x */
        if (incx == 1) {
            for (int i = 0; i < m; i++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    sum += A[i + j * lda] * x[j];
                }
                y[i * incy] += alpha * sum;
            }
        } else {
            for (int i = 0; i < m; i++) {
                float sum = 0.0f;
                int jx = 0;
                for (int j = 0; j < n; j++) {
                    sum += A[i + j * lda] * x[jx];
                    jx += incx;
                }
                y[i * incy] += alpha * sum;
            }
        }
    } else {
        /* y := alpha*A^T*x + beta*y (or A^H for complex) */
        
        /* Scale y by beta */
        if (beta == 0.0f) {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                y[jy] = 0.0f;
                jy += incy;
            }
        } else if (beta != 1.0f) {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                y[jy] *= beta;
                jy += incy;
            }
        }
        
        if (alpha == 0.0f) return;
        
        /* Compute alpha*A^T*x */
        if (incx == 1) {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int i = 0; i < m; i++) {
                    sum += A[i + j * lda] * x[i];
                }
                y[jy] += alpha * sum;
                jy += incy;
            }
        } else {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                int ix = 0;
                for (int i = 0; i < m; i++) {
                    sum += A[i + j * lda] * x[ix];
                    ix += incx;
                }
                y[jy] += alpha * sum;
                jy += incy;
            }
        }
    }
}
