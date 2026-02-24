/**
 * @file dsyr2k.c
 * @brief Reference implementation of DSYR2K (Double precision symmetric rank-2k update)
 */

#include <blas_reference.h>

void dsyr2k_ref(char uplo, char trans, int n, int k, double alpha,
                const double *A, int lda, const double *B, int ldb,
                double beta, double *C, int ldc) {
    
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
    
    for (int j = 0; j < n; j++) {
        int start = is_lower ? j : 0;
        int end = is_lower ? n : j + 1;
        for (int i = start; i < end; i++) {
            C[i + j * ldc] *= beta;
        }
    }
    
    if (is_trans) {
        for (int i = 0; i < n; i++) {
            for (int j = (is_lower ? i : 0); j < (is_lower ? n : i + 1); j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i + l * lda] * B[j + l * ldb];
                    sum += B[i + l * ldb] * A[j + l * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            for (int j = (is_lower ? i : 0); j < (is_lower ? n : i + 1); j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i + l * lda] * B[j + l * ldb];
                    sum += B[i + l * ldb] * A[j + l * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    }
}
