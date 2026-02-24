/**
 * @file dsymm.c
 * @brief Reference implementation of DSYMM (Double precision symmetric matrix multiply)
 *
 * DSYMM: C := alpha*A*B + beta*C (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision symmetric matrix multiply
 */
void dsymm_ref(char side, char uplo, int m, int n, double alpha,
               const double *A, int lda, const double *B, int ldb,
               double beta, double *C, int ldc) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0 && beta == 1.0) return;
    
    int is_left = (side == 'L' || side == 'l');
    
    for (int j = 0; j < n; j++) {
        if (beta == 0.0) {
            for (int i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0;
            }
        } else if (beta != 1.0) {
            for (int i = 0; i < m; i++) {
                C[i + j * ldc] *= beta;
            }
        }
    }
    
    if (alpha == 0.0) return;
    
    if (is_left) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int k = 0; k < m; k++) {
                    sum += A[i + k * lda] * B[k + j * ldb];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += B[i + k * ldb] * A[k + j * lda];
                }
                C[i + j * ldc] += alpha * sum;
            }
        }
    }
}
