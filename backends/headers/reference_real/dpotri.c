#
#include <blas_reference.h>

void dpotri_(const char *uplo, const int *n, double *A, const int *lda, int *info)
{
    *info = 0;
    if (*n <= 0) return;
    
    // A now contains L or U from POTRF
    // Compute inv(L)*inv(L)^T or inv(U)^T*inv(U)
    
    if (*uplo == 'U' || *uplo == 'u') {
        // Upper: process from bottom-right
        for (int j = *n - 1; j >= 0; j--) {
            if (A[j + j * *lda] == 0.0) { *info = j + 1; return; }
            
            A[j + j * *lda] = 1.0 / A[j + j * *lda];
            double ajj = -A[j + j * *lda];
            
            // Scale by diagonal
            for (int i = 0; i < j; i++)
                A[i + j * *lda] *= ajj;
            
            // Update upper triangle
            for (int k = 0; k < j; k++) {
                A[k + j * *lda] = A[k + j * *lda] / A[k + k * *lda];
                double akj = -A[k + j * *lda];
                for (int i = 0; i < k; i++)
                    A[i + j * *lda] += akj * A[i + k * *lda];
            }
        }
    } else {
        // Lower: process from top-left
        for (int j = 0; j < *n; j++) {
            if (A[j + j * *lda] == 0.0) { *info = j + 1; return; }
            
            A[j + j * *lda] = 1.0 / A[j + j * *lda];
            double ajj = -A[j + j * *lda];
            
            for (int i = j + 1; i < *n; i++)
                A[i + j * *lda] *= ajj;
            
            for (int k = j + 1; k < *n; k++) {
                A[k + j * *lda] = A[k + j * *lda] / A[k + k * *lda];
                double akj = -A[k + j * *lda];
                for (int i = k + 1; i < *n; i++)
                    A[i + j * *lda] += akj * A[i + k * *lda];
            }
        }
    }
}
