/**
 * @file sbmv.c
 * @brief Reference implementation of SBMV (Symmetric banded matrix-vector)
 *
 * SBMV: y := alpha*A*x + beta*y
 * where A is symmetric banded matrix stored in compact form
 */

#include <blas_reference.h>

void sbmv_ref(char uplo, int n, int k, float alpha, const float *A, int lda,
              const float *x, int incx, float beta, float *y, int incy) {
    
    if (n <= 0 || (alpha == 0.0f && beta == 1.0f)) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    /* Scale y */
    if (beta != 1.0f) {
        int iy = (incy > 0 ? 0 : (n - 1) * (-incy));
        if (beta == 0.0f) {
            for (int i = 0; i < n; i++) {
                y[iy] = 0.0f;
                iy += incy;
            }
        } else {
            for (int i = 0; i < n; i++) {
                y[iy] *= beta;
                iy += incy;
            }
        }
    }
    
    /* Matrix-vector product */
    if (is_lower) {
        /* Lower triangular banded: A[i,j] stored at A[k + i - j + j*lda] for i >= j */
        int ix = 0;
        for (int i = 0; i < n; i++) {
            float xi = alpha * x[ix];
            
            /* Diagonal element */
            y[i * incy] += xi * A[k + i * lda];
            
            /* Off-diagonal elements below diagonal */
            int jmin = (i + 1 < n) ? i + 1 : n;
            int jmax = (i + k < n) ? i + k : n - 1;
            for (int j = jmin; j <= jmax; j++) {
                int idx = k + j - i + i * lda;
                y[j * incy] += xi * A[idx];
                /* Symmetric: A[i,j] = A[j,i] */
                y[i * incy] += alpha * x[j * incx] * A[idx];
            }
            
            ix += incx;
        }
    } else {
        /* Upper triangular banded: A[i,j] stored at A[j - i + j*lda] for i <= j */
        for (int i = 0; i < n; i++) {
            float xi = alpha * x[i * incx];
            
            /* Diagonal element */
            y[i * incy] += xi * A[i * lda];
            
            /* Off-diagonal elements above diagonal */
            int jmin = (i - k >= 0) ? i - k : 0;
            for (int j = jmin; j < i; j++) {
                int idx = i - j + j * lda;
                y[j * incy] += xi * A[idx];
                /* Symmetric */
                y[i * incy] += alpha * x[j * incx] * A[idx];
            }
        }
    }
}
