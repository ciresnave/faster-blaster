/**
 * @file strmv.c
 * @brief Reference implementation of STRMV (Single precision triangular matrix-vector multiply)
 *
 * STRMV: x := A*x (or A^T*x) where A is triangular
 *
 * Similar to TRSV but does multiplication instead of solving
 */

#include <blas_reference.h>

/**
 * Single precision triangular matrix-vector multiply
 *
 * @param uplo 'U'/'L' - upper or lower triangular
 * @param trans 'N'/'T'/'C' - no transpose, transpose, conjugate transpose
 * @param diag 'U'/'N' - unit diagonal or non-unit
 * @param n Dimension of A
 * @param A Triangular matrix (n x n, column-major)
 * @param lda Leading dimension of A
 * @param x Vector x (in-out)
 * @param incx Stride for x
 */
void strmv_ref(char uplo, char trans, char diag, int n, const float *A, int lda,
               float *x, int incx) {
    
    if (n <= 0) return;
    
    int is_unit = (diag == 'U' || diag == 'u');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_notrans = (trans == 'N' || trans == 'n');
    
    float temp[1000];  /* Temporary storage for result */
    
    if (is_lower && is_notrans) {
        /* L*x: forward multiplication */
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            if (is_unit) sum = x[i * incx];
            
            for (int j = 0; j <= i; j++) {
                if (!is_unit || j < i) {
                    sum += A[i + j * lda] * x[j * incx];
                }
            }
            temp[i] = sum;
        }
    } else if (!is_lower && is_notrans) {
        /* U*x: backward multiplication */
        for (int i = n - 1; i >= 0; i--) {
            float sum = 0.0f;
            if (is_unit) sum = x[i * incx];
            
            for (int j = i; j < n; j++) {
                if (!is_unit || j > i) {
                    sum += A[i + j * lda] * x[j * incx];
                }
            }
            temp[i] = sum;
        }
    } else if (is_lower && !is_notrans) {
        /* L^T*x */
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            if (is_unit) sum = x[j * incx];
            
            for (int i = j; i < n; i++) {
                if (!is_unit || i > j) {
                    sum += A[i + j * lda] * x[i * incx];
                }
            }
            temp[j] = sum;
        }
    } else {
        /* U^T*x */
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            if (is_unit) sum = x[j * incx];
            
            for (int i = 0; i <= j; i++) {
                if (!is_unit || i < j) {
                    sum += A[i + j * lda] * x[i * incx];
                }
            }
            temp[j] = sum;
        }
    }
    
    /* Copy result back */
    for (int i = 0; i < n; i++) {
        x[i * incx] = temp[i];
    }
}
