/**
 * @file ssymv.c
 * @brief Reference implementation of SSYMV (Single precision symmetric matrix-vector multiply)
 *
 * SSYMV: y := alpha*A*x + beta*y where A is symmetric
 */

#include <blas_reference.h>

/**
 * Single precision symmetric matrix-vector multiply
 *
 * @param uplo 'U'/'L' - upper or lower triangle stored
 * @param n Dimension of A
 * @param alpha Scalar
 * @param A Symmetric matrix (n x n)
 * @param lda Leading dimension
 * @param x Vector x
 * @param incx Stride
 * @param beta Scalar
 * @param y Vector y (in-out)
 * @param incy Stride
 */
void ssymv_ref(char uplo, int n, float alpha, const float *A, int lda,
               const float *x, int incx, float beta, float *y, int incy) {
    
    if (n <= 0) return;
    if (alpha == 0.0f && beta == 1.0f) return;
    
    /* Scale y */
    if (beta == 0.0f) {
        int iy = 0;
        for (int i = 0; i < n; i++) {
            y[iy] = 0.0f;
            iy += incy;
        }
    } else if (beta != 1.0f) {
        int iy = 0;
        for (int i = 0; i < n; i++) {
            y[iy] *= beta;
            iy += incy;
        }
    }
    
    if (alpha == 0.0f) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    if (is_lower) {
        /* Lower triangle: A[i,j] stored for i >= j */
        int jx = 0;
        int jy = 0;
        for (int j = 0; j < n; j++) {
            float xj = x[jx];
            float temp = 0.0f;
            
            int iy = 0;
            for (int i = j; i < n; i++) {
                y[iy] += alpha * A[i + j * lda] * xj;
                temp += A[i + j * lda] * x[iy];
                iy += incy;
            }
            
            if (j > 0) {
                int iy2 = 0;
                for (int i = 0; i < j; i++) {
                    y[iy2] += alpha * A[j + i * lda] * x[iy2];
                    iy2 += incy;
                }
            }
            
            y[jy] += alpha * temp;
            jx += incx;
            jy += incy;
        }
    } else {
        /* Upper triangle: A[i,j] stored for i <= j */
        int jx = 0;
        int jy = 0;
        for (int j = 0; j < n; j++) {
            float xj = x[jx];
            float temp = 0.0f;
            
            int iy = 0;
            for (int i = 0; i <= j; i++) {
                y[iy] += alpha * A[i + j * lda] * xj;
                temp += A[i + j * lda] * x[iy];
                iy += incy;
            }
            
            if (j < n - 1) {
                int iy2 = (j + 1) * incy;
                for (int i = j + 1; i < n; i++) {
                    y[iy2] += alpha * A[j + i * lda] * x[iy2];
                    iy2 += incy;
                }
            }
            
            y[jy] += alpha * temp;
            jx += incx;
            jy += incy;
        }
    }
}
