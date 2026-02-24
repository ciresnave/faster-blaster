/**
 * @file stbmv.c
 * @brief Reference implementation of STBMV (Triangular banded matrix-vector)
 *
 * STBMV: x := A*x or x := A^T*x
 * where A is triangular banded matrix
 */

#include <blas_reference.h>

void stbmv_ref(char uplo, char trans, char diag, int n, int k,
               const float *A, int lda, float *x, int incx) {
    
    if (n <= 0) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_notrans = (trans == 'N' || trans == 'n');
    int is_unit = (diag == 'U' || diag == 'u');
    
    if (is_notrans) {
        if (is_lower) {
            /* Forward sweep: x := L*x */
            int ix = 0;
            for (int i = 0; i < n; i++) {
                float xi = x[ix];
                
                /* Off-diagonal */
                int jmin = (i - k >= 0) ? i - k : 0;
                for (int j = jmin; j < i; j++) {
                    int idx = i - j + j * lda;
                    xi += A[idx] * x[j * incx];
                }
                
                /* Diagonal */
                if (!is_unit) {
                    xi /= A[k + i * lda];
                }
                x[ix] = xi;
                
                ix += incx;
            }
        } else {
            /* Backward sweep: x := U*x */
            int ix = (n - 1) * incx;
            for (int i = n - 1; i >= 0; i--) {
                float xi = x[ix];
                
                /* Off-diagonal */
                int jmax = (i + k < n) ? i + k : n - 1;
                for (int j = i + 1; j <= jmax; j++) {
                    int idx = j - i + i * lda;
                    xi += A[idx] * x[j * incx];
                }
                
                /* Diagonal */
                if (!is_unit) {
                    xi /= A[i * lda];
                }
                x[ix] = xi;
                
                ix -= incx;
            }
        }
    } else {
        if (is_lower) {
            /* Backward sweep: x := L^T*x */
            int ix = (n - 1) * incx;
            for (int i = n - 1; i >= 0; i--) {
                float xi = x[ix];
                
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
            /* Forward sweep: x := U^T*x */
            int ix = 0;
            for (int i = 0; i < n; i++) {
                float xi = x[ix];
                
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
