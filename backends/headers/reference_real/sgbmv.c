/**
 * @file sgbmv.c
 * @brief Reference implementation of SGBMV (General banded matrix-vector multiply)
 *
 * SGBMV: y := alpha*A*x + beta*y
 * where A is m-by-n banded matrix stored in compact form
 */

#include <blas_reference.h>

void sgbmv_ref(char trans, int m, int n, int kl, int ku, float alpha,
               const float *A, int lda, const float *x, int incx,
               float beta, float *y, int incy) {
    
    if (m <= 0 || n <= 0 || (alpha == 0.0f && beta == 1.0f)) return;
    
    int is_notrans = (trans == 'N' || trans == 'n');
    
    if (is_notrans) {
        /* y := alpha*A*x + beta*y */
        int nrow = m;
        int ncol = n;
        
        /* Scale y */
        if (beta != 1.0f) {
            int iy = (beta == 0.0f) ? 0 : (incy > 0 ? 0 : (nrow - 1) * (-incy));
            if (beta == 0.0f) {
                for (int i = 0; i < nrow; i++) {
                    y[iy] = 0.0f;
                    iy += incy;
                }
            } else {
                for (int i = 0; i < nrow; i++) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
        
        /* Matrix-vector product */
        for (int j = 0; j < ncol; j++) {
            float xj = alpha * x[j * incx];
            int k_start = (j - ku < 0) ? 0 : j - ku;
            int k_end = (j + kl >= nrow) ? nrow : j + kl + 1;
            
            for (int i = k_start; i < k_end; i++) {
                /* A[i,j] stored at A[ku + i - j + j*lda] */
                int idx = ku + i - j + j * lda;
                y[i * incy] += xj * A[idx];
            }
        }
    } else {
        /* y := alpha*A^T*x + beta*y (or A^H for complex) */
        int nrow = n;
        int ncol = m;
        
        if (beta != 1.0f) {
            int iy = (incy > 0 ? 0 : (nrow - 1) * (-incy));
            if (beta == 0.0f) {
                for (int i = 0; i < nrow; i++) {
                    y[iy] = 0.0f;
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
            float xi = alpha * x[i * incx];
            int k_start = (i - kl < 0) ? 0 : i - kl;
            int k_end = (i + ku >= nrow) ? nrow : i + ku + 1;
            
            for (int j = k_start; j < k_end; j++) {
                int idx = ku + i - j + j * lda;
                y[j * incy] += xi * A[idx];
            }
        }
    }
}
