/**
 * @file dgemv.c
 * @brief Reference implementation of DGEMV (Double precision general matrix-vector multiply)
 *
 * DGEMV: y := alpha*A*x + beta*y (or A^T*x)
 */

#include <blas_reference.h>

/**
 * Double precision general matrix-vector multiply
 */
void dgemv_ref(char trans, int m, int n, double alpha, const double *A, int lda,
               const double *x, int incx, double beta, double *y, int incy) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0 && beta == 1.0) return;
    
    if (trans == 'N' || trans == 'n') {
        /* y := alpha*A*x + beta*y */
        
        if (beta == 0.0) {
            int iy = 0;
            for (int i = 0; i < m; i++) {
                y[iy] = 0.0;
                iy += incy;
            }
        } else if (beta != 1.0) {
            int iy = 0;
            for (int i = 0; i < m; i++) {
                y[iy] *= beta;
                iy += incy;
            }
        }
        
        if (alpha == 0.0) return;
        
        if (incx == 1) {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += A[i + j * lda] * x[j];
                }
                y[i * incy] += alpha * sum;
            }
        } else {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                int jx = 0;
                for (int j = 0; j < n; j++) {
                    sum += A[i + j * lda] * x[jx];
                    jx += incx;
                }
                y[i * incy] += alpha * sum;
            }
        }
    } else {
        /* y := alpha*A^T*x + beta*y */
        
        if (beta == 0.0) {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                y[jy] = 0.0;
                jy += incy;
            }
        } else if (beta != 1.0) {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                y[jy] *= beta;
                jy += incy;
            }
        }
        
        if (alpha == 0.0) return;
        
        if (incx == 1) {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < m; i++) {
                    sum += A[i + j * lda] * x[i];
                }
                y[jy] += alpha * sum;
                jy += incy;
            }
        } else {
            int jy = 0;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
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
