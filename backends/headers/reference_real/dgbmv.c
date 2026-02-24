/**
 * @file dgbmv.c
 * @brief Reference implementation of DGBMV (Double precision banded matrix-vector)
 */

#include <blas_reference.h>

void dgbmv_ref(char trans, int m, int n, int kl, int ku, double alpha,
               const double *A, int lda, const double *x, int incx,
               double beta, double *y, int incy) {
    
    if (m <= 0 || n <= 0 || (alpha == 0.0 && beta == 1.0)) return;
    
    int is_notrans = (trans == 'N' || trans == 'n');
    
    if (is_notrans) {
        int nrow = m;
        int ncol = n;
        
        if (beta != 1.0) {
            int iy = (incy > 0 ? 0 : (nrow - 1) * (-incy));
            if (beta == 0.0) {
                for (int i = 0; i < nrow; i++) {
                    y[iy] = 0.0;
                    iy += incy;
                }
            } else {
                for (int i = 0; i < nrow; i++) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
        
        for (int j = 0; j < ncol; j++) {
            double xj = alpha * x[j * incx];
            int k_start = (j - ku < 0) ? 0 : j - ku;
            int k_end = (j + kl >= nrow) ? nrow : j + kl + 1;
            
            for (int i = k_start; i < k_end; i++) {
                int idx = ku + i - j + j * lda;
                y[i * incy] += xj * A[idx];
            }
        }
    } else {
        int nrow = n;
        int ncol = m;
        
        if (beta != 1.0) {
            int iy = (incy > 0 ? 0 : (nrow - 1) * (-incy));
            if (beta == 0.0) {
                for (int i = 0; i < nrow; i++) {
                    y[iy] = 0.0;
                    iy += incy;
                }
            } else {
                for (int i = 0; i < nrow; i++) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
        
        for (int i = 0; i < ncol; i++) {
            double xi = alpha * x[i * incx];
            int k_start = (i - kl < 0) ? 0 : i - kl;
            int k_end = (i + ku >= nrow) ? nrow : i + ku + 1;
            
            for (int j = k_start; j < k_end; j++) {
                int idx = ku + i - j + j * lda;
                y[j * incy] += xi * A[idx];
            }
        }
    }
}
