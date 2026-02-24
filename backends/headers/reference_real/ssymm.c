/**
 * @file ssymm.c
 * @brief Reference implementation of SSYMM (Single precision symmetric matrix multiply)
 *
 * SSYMM: C := alpha*A*B + beta*C (where A is symmetric) or C := alpha*B*A + beta*C
 */

#include <blas_reference.h>

/**
 * Single precision symmetric matrix multiply
 *
 * @param side 'L'/'R' - A on left or right
 * @param uplo 'U'/'L' - upper or lower triangle of A
 * @param m Number of rows in C
 * @param n Number of columns in C
 * @param alpha Scalar
 * @param A Symmetric matrix (m x m or n x n depending on side)
 * @param lda Leading dimension of A
 * @param B Matrix B (m x n)
 * @param ldb Leading dimension of B
 * @param beta Scalar
 * @param C Matrix C (m x n, in-out)
 * @param ldc Leading dimension of C
 */
void ssymm_ref(char side, char uplo, int m, int n, float alpha,
               const float *A, int lda, const float *B, int ldb,
               float beta, float *C, int ldc) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0f && beta == 1.0f) return;
    
    int is_left = (side == 'L' || side == 'l');
    
    /* Scale C by beta */
    for (int j = 0; j < n; j++) {
        if (beta == 0.0f) {
            for (int i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0f;
            }
        } else if (beta != 1.0f) {
            for (int i = 0; i < m; i++) {
                C[i + j * ldc] *= beta;
            }
        }
    }
    
    if (alpha == 0.0f) return;
    
    if (is_left) {
        /* C := alpha*A*B + beta*C (A is m x m) */
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                float sum = 0.0f;
                for (int k = 0; k < m; k++) {
                    sum += A[i + k * lda] * B[k + j * ldb];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    } else {
        /* C := alpha*B*A + beta*C (A is n x n) */
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += B[i + k * ldb] * A[k + j * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    }
}
