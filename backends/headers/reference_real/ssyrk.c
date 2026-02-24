/**
 * @file ssyrk.c
 * @brief Reference implementation of SSYRK (Single precision symmetric rank-k update)
 *
 * SSYRK: C := alpha*A*A^T + beta*C or C := alpha*A^T*A + beta*C where C is symmetric
 *
 * C is symmetric, only upper or lower triangle is used/updated
 */

#include <blas_reference.h>

void ssyrk_ref(char uplo, char trans, int n, int k, float alpha,
               const float *A, int lda, float beta, float *C, int ldc) {
    
    if (n <= 0) return;
    if (k <= 0) {
        /* Beta scaling only (no update) */
        int is_lower = (uplo == 'L' || uplo == 'l');
        for (int j = 0; j < n; j++) {
            int start = is_lower ? j : 0;
            int end = is_lower ? n : j + 1;
            for (int i = start; i < end; i++) {
                C[i + j * ldc] *= beta;
            }
        }
        return;
    }
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_trans = (trans == 'T' || trans == 't');
    
    /* Beta scaling of C */
    for (int j = 0; j < n; j++) {
        int start = is_lower ? j : 0;
        int end = is_lower ? n : j + 1;
        for (int i = start; i < end; i++) {
            C[i + j * ldc] *= beta;
        }
    }
    
    /* Alpha update */
    if (is_trans) {
        /* C := alpha*A^T*A + beta*C */
        for (int i = 0; i < n; i++) {
            for (int j = (is_lower ? i : 0); j < (is_lower ? n : i + 1); j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[i + l * lda] * A[j + l * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    } else {
        /* C := alpha*A*A^T + beta*C */
        for (int i = 0; i < n; i++) {
            for (int j = (is_lower ? i : 0); j < (is_lower ? n : i + 1); j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[i + l * lda] * A[j + l * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    }
}
