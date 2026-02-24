/**
 * @file strsv.c
 * @brief Reference implementation of STRSV (Single precision triangular solve)
 *
 * STRSV: x := A^(-1)*x (or A^T*x = b for triangular A)
 *
 * Solves the triangular system A*x = b where A is triangular
 * Uses forward/backward substitution
 */

#include <blas_reference.h>

/**
 * Single precision triangular solve
 *
 * @param uplo 'U'/'L' - upper or lower triangular
 * @param trans 'N'/'T'/'C' - no transpose, transpose, conjugate transpose
 * @param diag 'U'/'N' - unit diagonal or non-unit
 * @param n Dimension of A
 * @param A Triangular matrix (n x n, column-major)
 * @param lda Leading dimension of A
 * @param x Vector x (will be overwritten with solution)
 * @param incx Stride for x
 */
void strsv_ref(char uplo, char trans, char diag, int n, const float *A, int lda,
               float *x, int incx) {
    
    if (n <= 0) return;
    
    int is_unit = (diag == 'U' || diag == 'u');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_notrans = (trans == 'N' || trans == 'n');
    
    if (is_lower && is_notrans) {
        /* Lower triangular, no transpose: forward substitution
         * A*x = b => solve L*x = b
         */
        int ix = 0;
        for (int i = 0; i < n; i++) {
            float sum = x[ix];
            int jx = 0;
            for (int j = 0; j < i; j++) {
                sum -= A[i + j * lda] * x[jx];
                jx += incx;
            }
            
            if (!is_unit) {
                float aii = A[i + i * lda];
                if (aii == 0.0f) aii = 1.0f;  /* Avoid division by zero */
                sum /= aii;
            }
            x[ix] = sum;
            ix += incx;
        }
    } else if (!is_lower && is_notrans) {
        /* Upper triangular, no transpose: backward substitution
         * U*x = b
         */
        int ix = (n - 1) * incx;
        for (int i = n - 1; i >= 0; i--) {
            float sum = x[ix];
            int jx = n * incx;
            for (int j = n - 1; j > i; j--) {
                jx -= incx;
                sum -= A[i + j * lda] * x[jx];
            }
            
            if (!is_unit) {
                float aii = A[i + i * lda];
                if (aii == 0.0f) aii = 1.0f;
                sum /= aii;
            }
            x[ix] = sum;
            ix -= incx;
        }
    } else if (is_lower && !is_notrans) {
        /* Lower triangular, transposed */
        int ix = (n - 1) * incx;
        for (int j = n - 1; j >= 0; j--) {
            float sum = x[ix];
            int ix_temp = 0;
            for (int i = 0; i < j; i++) {
                sum -= A[j + i * lda] * x[ix_temp];
                ix_temp += incx;
            }
            
            if (!is_unit) {
                float ajj = A[j + j * lda];
                if (ajj == 0.0f) ajj = 1.0f;
                sum /= ajj;
            }
            x[ix] = sum;
            ix -= incx;
        }
    } else {
        /* Upper triangular, transposed */
        int ix = 0;
        for (int j = 0; j < n; j++) {
            float sum = x[ix];
            int ix_temp = (j + 1) * incx;
            for (int i = j + 1; i < n; i++) {
                sum -= A[j + i * lda] * x[ix_temp];
                ix_temp += incx;
            }
            
            if (!is_unit) {
                float ajj = A[j + j * lda];
                if (ajj == 0.0f) ajj = 1.0f;
                sum /= ajj;
            }
            x[ix] = sum;
            ix += incx;
        }
    }
}
