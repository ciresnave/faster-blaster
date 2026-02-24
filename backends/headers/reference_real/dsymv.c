/**
 * @file dsymv.c
 * @brief Reference implementation of DSYMV (Double precision symmetric matrix-vector multiply)
 *
 * DSYMV: y := alpha*A*x + beta*y (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision symmetric matrix-vector multiply
 */
void dsymv_ref(char uplo, int n, double alpha, const double *A, int lda,
               const double *x, int incx, double beta, double *y, int incy) {
    
    if (n <= 0) return;
    if (alpha == 0.0 && beta == 1.0) return;
    
    if (beta == 0.0) {
        int iy = 0;
        for (int i = 0; i < n; i++) {
            y[iy] = 0.0;
            iy += incy;
        }
    } else if (beta != 1.0) {
        int iy = 0;
        for (int i = 0; i < n; i++) {
            y[iy] *= beta;
            iy += incy;
        }
    }
    
    if (alpha == 0.0) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    if (is_lower) {
        int jx = 0;
        int jy = 0;
        for (int j = 0; j < n; j++) {
            double xj = x[jx];
            double temp = 0.0;
            
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
        int jx = 0;
        int jy = 0;
        for (int j = 0; j < n; j++) {
            double xj = x[jx];
            double temp = 0.0;
            
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
