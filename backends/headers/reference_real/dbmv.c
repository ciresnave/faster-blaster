/**
 * @file dbmv.c
 * @brief Reference implementation of DBMV (Double precision symmetric banded)
 */

#include <blas_reference.h>

void dbmv_ref(char uplo, int n, int k, double alpha, const double *A, int lda,
              const double *x, int incx, double beta, double *y, int incy) {
    
    if (n <= 0 || (alpha == 0.0 && beta == 1.0)) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    if (beta != 1.0) {
        int iy = (incy > 0 ? 0 : (n - 1) * (-incy));
        if (beta == 0.0) {
            for (int i = 0; i < n; i++) {
                y[iy] = 0.0;
                iy += incy;
            }
        } else {
            for (int i = 0; i < n; i++) {
                y[iy] *= beta;
                iy += incy;
            }
        }
    }
    
    if (is_lower) {
        int ix = 0;
        for (int i = 0; i < n; i++) {
            double xi = alpha * x[ix];
            
            y[i * incy] += xi * A[k + i * lda];
            
            int jmin = (i + 1 < n) ? i + 1 : n;
            int jmax = (i + k < n) ? i + k : n - 1;
            for (int j = jmin; j <= jmax; j++) {
                int idx = k + j - i + i * lda;
                y[j * incy] += xi * A[idx];
                y[i * incy] += alpha * x[j * incx] * A[idx];
            }
            
            ix += incx;
        }
    } else {
        for (int i = 0; i < n; i++) {
            double xi = alpha * x[i * incx];
            
            y[i * incy] += xi * A[i * lda];
            
            int jmin = (i - k >= 0) ? i - k : 0;
            for (int j = jmin; j < i; j++) {
                int idx = i - j + j * lda;
                y[j * incy] += xi * A[idx];
                y[i * incy] += alpha * x[j * incx] * A[idx];
            }
        }
    }
}
