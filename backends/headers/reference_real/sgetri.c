/**
 * @file sgetri.c
 * @brief Reference implementation of SGETRI (LU-based matrix inversion)
 *
 * SGETRI computes the inverse of a matrix using the LU factorization
 * computed by SGETRF. This is a simplified reference implementation
 * using sequential updates (not optimized for cache).
 */

#include <blas_reference.h>
#include <string.h>

/* Static helper: Invert triangular matrix in-place */
static void strtri_helper(char uplo, char diag, int n, float *A, int lda, int *info) {
    
    if (n <= 0) {
        *info = 0;
        return;
    }
    
    *info = 0;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_unit = (diag == 'U' || diag == 'u');
    
    if (is_lower) {
        /* Invert lower triangular */
        for (int j = 0; j < n; j++) {
            if (!is_unit) {
                if (A[j + j * lda] == 0.0f) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda] = 1.0f / A[j + j * lda];
            }
            float ajj = A[j + j * lda];
            
            for (int i = j + 1; i < n; i++) {
                float sum = 0.0f;
                for (int k = j; k < i; k++) {
                    sum -= A[i + k * lda] * A[k + j * lda];
                }
                A[i + j * lda] = sum / ajj;
            }
        }
    } else {
        /* Invert upper triangular */
        for (int j = n - 1; j >= 0; j--) {
            if (!is_unit) {
                if (A[j + j * lda] == 0.0f) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda] = 1.0f / A[j + j * lda];
            }
            float ajj = A[j + j * lda];
            
            for (int i = j - 1; i >= 0; i--) {
                float sum = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    sum -= A[i + k * lda] * A[k + j * lda];
                }
                A[i + j * lda] = sum / ajj;
            }
        }
    }
}

void sgetri_(const int *n, float *A, const int *lda, const int *ipiv, int *info)
{
    int nn = *n;
    int lda_val = *lda;
    
    if (nn <= 0) {
        *info = 0;
        return;
    }
    
    *info = 0;
    
    /* Invert the upper triangular matrix U from LU factorization */
    strtri_helper('U', 'N', nn, A, lda_val, info);
    
    if (*info != 0) return;
    
    /* Solve L*X = I to get inverse */
    /* Working backward from the LU factorization */
    for (int k = nn - 2; k >= 0; k--) {
        /* Compute L[k:n, k:n]^(-1) and update rest */
        for (int i = k + 1; i < nn; i++) {
            for (int j = 0; j < nn; j++) {
                A[i + j * lda_val] -= A[k + j * lda_val] * A[i + k * lda_val];
            }
        }
    }
    
    /* Apply row permutations in reverse order */
    for (int i = nn - 1; i >= 0; i--) {
        int ipiv_i = ipiv[i] - 1;  /* Convert from 1-based to 0-based */
        if (ipiv_i != i) {
            /* Swap rows i and ipiv_i */
            for (int j = 0; j < nn; j++) {
                float temp = A[i + j * lda_val];
                A[i + j * lda_val] = A[ipiv_i + j * lda_val];
                A[ipiv_i + j * lda_val] = temp;
            }
        }
    }
}
