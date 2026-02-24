/**
 * @file strsm.c
 * @brief Reference implementation of STRSM (Single precision triangular solve with matrix RHS)
 *
 * STRSM: B := alpha*A^(-1)*B (or B := alpha*B*A^(-1)) where A is triangular
 */

#include <blas_reference.h>

/**
 * Single precision triangular solve with matrix RHS
 *
 * @param side 'L'/'R' - A on left or right
 * @param uplo 'U'/'L' - upper or lower triangle
 * @param trans 'N'/'T'/'C' - no transpose, transpose, conjugate transpose
 * @param diag 'U'/'N' - unit diagonal or non-unit
 * @param m Number of rows in B
 * @param n Number of columns in B
 * @param alpha Scalar
 * @param A Triangular matrix (m x m or n x n)
 * @param lda Leading dimension
 * @param B Matrix B (m x n, in-out, solution on exit)
 * @param ldb Leading dimension
 */
void strsm_ref(char side, char uplo, char trans, char diag, int m, int n, float alpha,
               const float *A, int lda, float *B, int ldb) {
    
    if (m <= 0 || n <= 0) return;
    
    /* Scale B by alpha */
    if (alpha != 1.0f) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                B[i + j * ldb] *= alpha;
            }
        }
    }
    
    int is_left = (side == 'L' || side == 'l');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_unit = (diag == 'U' || diag == 'u');
    
    if (is_left) {
        /* B := A^(-1)*B (A is m x m) */
        for (int j = 0; j < n; j++) {
            if (is_lower) {
                /* Forward substitution */
                for (int i = 0; i < m; i++) {
                    float sum = B[i + j * ldb];
                    
                    for (int k = 0; k < i; k++) {
                        sum -= A[i + k * lda] * B[k + j * ldb];
                    }
                    
                    if (!is_unit) {
                        float aii = A[i + i * lda];
                        if (aii != 0.0f) sum /= aii;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            } else {
                /* Backward substitution */
                for (int i = m - 1; i >= 0; i--) {
                    float sum = B[i + j * ldb];
                    
                    for (int k = i + 1; k < m; k++) {
                        sum -= A[i + k * lda] * B[k + j * ldb];
                    }
                    
                    if (!is_unit) {
                        float aii = A[i + i * lda];
                        if (aii != 0.0f) sum /= aii;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            }
        }
    } else {
        /* B := B*A^(-1) (A is n x n) */
        if (is_lower) {
            /* Backward substitution on columns */
            for (int j = n - 1; j >= 0; j--) {
                for (int i = 0; i < m; i++) {
                    float sum = B[i + j * ldb];
                    
                    for (int k = j + 1; k < n; k++) {
                        sum -= B[i + k * ldb] * A[j + k * lda];
                    }
                    
                    if (!is_unit) {
                        float ajj = A[j + j * lda];
                        if (ajj != 0.0f) sum /= ajj;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            }
        } else {
            /* Forward substitution on columns */
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    float sum = B[i + j * ldb];
                    
                    for (int k = 0; k < j; k++) {
                        sum -= B[i + k * ldb] * A[k + j * lda];
                    }
                    
                    if (!is_unit) {
                        float ajj = A[j + j * lda];
                        if (ajj != 0.0f) sum /= ajj;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            }
        }
    }
}
