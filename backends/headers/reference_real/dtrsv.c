/**
 * @file dtrsv.c
 * @brief Reference implementation of DTRSV (Double precision triangular solve)
 *
 * DTRSV: x := A^(-1)*x (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision triangular solve
 */
void dtrsv_ref(char uplo, char trans, char diag, int n, const double *A, int lda,
               double *x, int incx) {
    
    if (n <= 0) return;
    
    int is_unit = (diag == 'U' || diag == 'u');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_notrans = (trans == 'N' || trans == 'n');
    
    if (is_lower && is_notrans) {
        int ix = 0;
        for (int i = 0; i < n; i++) {
            double sum = x[ix];
            int jx = 0;
            for (int j = 0; j < i; j++) {
                sum -= A[i + j * lda] * x[jx];
                jx += incx;
            }
            
            if (!is_unit) {
                double aii = A[i + i * lda];
                if (aii == 0.0) aii = 1.0;
                sum /= aii;
            }
            x[ix] = sum;
            ix += incx;
        }
    } else if (!is_lower && is_notrans) {
        int ix = (n - 1) * incx;
        for (int i = n - 1; i >= 0; i--) {
            double sum = x[ix];
            int jx = n * incx;
            for (int j = n - 1; j > i; j--) {
                jx -= incx;
                sum -= A[i + j * lda] * x[jx];
            }
            
            if (!is_unit) {
                double aii = A[i + i * lda];
                if (aii == 0.0) aii = 1.0;
                sum /= aii;
            }
            x[ix] = sum;
            ix -= incx;
        }
    } else if (is_lower && !is_notrans) {
        int ix = (n - 1) * incx;
        for (int j = n - 1; j >= 0; j--) {
            double sum = x[ix];
            int ix_temp = 0;
            for (int i = 0; i < j; i++) {
                sum -= A[j + i * lda] * x[ix_temp];
                ix_temp += incx;
            }
            
            if (!is_unit) {
                double ajj = A[j + j * lda];
                if (ajj == 0.0) ajj = 1.0;
                sum /= ajj;
            }
            x[ix] = sum;
            ix -= incx;
        }
    } else {
        int ix = 0;
        for (int j = 0; j < n; j++) {
            double sum = x[ix];
            int ix_temp = (j + 1) * incx;
            for (int i = j + 1; i < n; i++) {
                sum -= A[j + i * lda] * x[ix_temp];
                ix_temp += incx;
            }
            
            if (!is_unit) {
                double ajj = A[j + j * lda];
                if (ajj == 0.0) ajj = 1.0;
                sum /= ajj;
            }
            x[ix] = sum;
            ix += incx;
        }
    }
}
