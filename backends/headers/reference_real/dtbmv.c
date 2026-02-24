/**
 * @file dtbmv.c
 * @brief Reference implementation of DTBMV (Double precision triangular banded)
 */

#include <blas_reference.h>

void dtbmv_ref(char uplo, char trans, char diag, int n, int k,
               const double *A, int lda, double *x, int incx) {
    
    if (n <= 0) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_notrans = (trans == 'N' || trans == 'n');
    int is_unit = (diag == 'U' || diag == 'u');
    
    if (is_notrans) {
        if (is_lower) {
            int ix = 0;
            for (int i = 0; i < n; i++) {
                double xi = x[ix];
                
                int jmin = (i - k >= 0) ? i - k : 0;
                for (int j = jmin; j < i; j++) {
                    int idx = i - j + j * lda;
                    xi += A[idx] * x[j * incx];
                }
                
                if (!is_unit) {
                    xi /= A[k + i * lda];
                }
                x[ix] = xi;
                
                ix += incx;
            }
        } else {
            int ix = (n - 1) * incx;
            for (int i = n - 1; i >= 0; i--) {
                double xi = x[ix];
                
                int jmax = (i + k < n) ? i + k : n - 1;
                for (int j = i + 1; j <= jmax; j++) {
                    int idx = j - i + i * lda;
                    xi += A[idx] * x[j * incx];
                }
                
                if (!is_unit) {
                    xi /= A[i * lda];
                }
                x[ix] = xi;
                
                ix -= incx;
            }
        }
    } else {
        if (is_lower) {
            int ix = (n - 1) * incx;
            for (int i = n - 1; i >= 0; i--) {
                double xi = x[ix];
                
                if (!is_unit) {
                    xi /= A[k + i * lda];
                }
                
                int jmin = (i - k >= 0) ? i - k : 0;
                for (int j = jmin; j < i; j++) {
                    int idx = i - j + j * lda;
                    x[j * incx] += xi * A[idx];
                }
                
                x[ix] = xi;
                ix -= incx;
            }
        } else {
            int ix = 0;
            for (int i = 0; i < n; i++) {
                double xi = x[ix];
                
                if (!is_unit) {
                    xi /= A[i * lda];
                }
                
                int jmax = (i + k < n) ? i + k : n - 1;
                for (int j = i + 1; j <= jmax; j++) {
                    int idx = j - i + i * lda;
                    x[j * incx] += xi * A[idx];
                }
                
                x[ix] = xi;
                ix += incx;
            }
        }
    }
}
