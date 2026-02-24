/**
 * @file ssyr2k.c
 * @brief Reference implementation of SSYR2K (Symmetric rank-2k update)
 *
 * SSYR2K: C := alpha*A*B^T + alpha*B*A^T + beta*C where C is symmetric
 */

#include <blas_reference.h>

void ssyr2k_ref(char uplo, char trans, int n, int k, float alpha,
                const float *A, int lda, const float *B, int ldb,
                float beta, float *C, int ldc) {
    
    if (n <= 0) return;
    if (k <= 0) {
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
        /* C := alpha*A^T*B + alpha*B^T*A + beta*C */
        for (int i = 0; i < n; i++) {
            for (int j = (is_lower ? i : 0); j < (is_lower ? n : i + 1); j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[i + l * lda] * B[j + l * ldb];
                    sum += B[i + l * ldb] * A[j + l * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    } else {
        /* C := alpha*A*B^T + alpha*B*A^T + beta*C */
        for (int i = 0; i < n; i++) {
            for (int j = (is_lower ? i : 0); j < (is_lower ? n : i + 1); j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[i + l * lda] * B[j + l * ldb];
                    sum += B[i + l * ldb] * A[j + l * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    }
}
