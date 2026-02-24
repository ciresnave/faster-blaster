/**
 * @file strmm.c
 * @brief Reference implementation of STRMM (Single precision triangular matrix multiply)
 *
 * STRMM: B := alpha*A*B (or B := alpha*B*A) where A is triangular
 */

#include <blas_reference.h>

/**
 * Single precision triangular matrix multiply
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
 * @param B Matrix B (m x n, in-out)
 * @param ldb Leading dimension
 */
void strmm_ref(char side, char uplo, char trans, char diag, int m, int n, float alpha,
               const float *A, int lda, float *B, int ldb) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0f) {
        /* Set B to zero */
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                B[i + j * ldb] = 0.0f;
            }
        }
        return;
    }
    
    int is_left = (side == 'L' || side == 'l');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_unit = (diag == 'U' || diag == 'u');
    int is_notrans = (trans == 'N' || trans == 'n');
    
    if (is_left) {
        /* B := alpha*A*B (A is m x m) */
        if (is_notrans) {
            for (int j = 0; j < n; j++) {
                for (int i = m - 1; i >= 0; i--) {
                    float sum = alpha * B[i + j * ldb];
                    
                    if (is_lower) {
                        for (int k = i + 1; k < m; k++) {
                            sum += A[i + k * lda] * B[k + j * ldb];
                        }
                    } else {
                        for (int k = 0; k < i; k++) {
                            sum += A[i + k * lda] * B[k + j * ldb];
                        }
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
        /* B := alpha*B*A (A is n x n) */
        for (int j = 0; j < n; j++) {
            if (alpha != 1.0f) {
                for (int i = 0; i < m; i++) {
                    B[i + j * ldb] *= alpha;
                }
            }
            
            if (is_lower) {
                for (int k = 0; k < j; k++) {
                    float akj = A[k + j * lda];
                    if (akj != 0.0f) {
                        for (int i = 0; i < m; i++) {
                            B[i + j * ldb] += akj * B[i + k * ldb];
                        }
                    }
                }
            } else {
                for (int k = j + 1; k < n; k++) {
                    float akj = A[k + j * lda];
                    if (akj != 0.0f) {
                        for (int i = 0; i < m; i++) {
                            B[i + j * ldb] += akj * B[i + k * ldb];
                        }
                    }
                }
            }
        }
    }
}
