/**
 * @file sormqr.c
 * @brief Reference implementation of SORMQR (Apply orthogonal matrix from QR)
 *
 * SORMQR multiplies an arbitrary real m-by-n matrix C by the orthogonal
 * matrix Q of a QR factorization formed from sgeqrf.
 *
 * Computes one of the matrix products:
 *    C := Q*C,   Q^T*C,   C*Q,  or  C*Q^T
 */

#

void sormqr_(const char *side, const char *trans, const int *m, const int *n, 
             const int *k, const float *A, const int *lda, const float *tau,
             float *C, const int *ldc, float *work, const int *lwork, int *info)
{
    int i, j, l;
    float sum, beta;
    int is_left = (side[0] == 'L' || side[0] == 'l');
    int is_notrans = (trans[0] == 'N' || trans[0] == 'n');
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    if (*k < 0) {
        *info = -5;
        return;
    }
    
    if (is_left) {
        /* Apply Q or Q^T from left: op(Q)*C */
        if (is_notrans) {
            /* Q*C: apply reflections forward */
            for (i = 0; i < *k; i++) {
                if (tau[i] == 0.0f) continue;
                
                /* Compute v^T*C where v = [1; A[i+1:m, i]] */
                for (j = 0; j < *n; j++) {
                    sum = C[i + j * *ldc];
                    for (l = i + 1; l < *m; l++) {
                        sum += A[l + i * *lda] * C[l + j * *ldc];
                    }
                    sum *= tau[i];
                    
                    C[i + j * *ldc] -= sum;
                    for (l = i + 1; l < *m; l++) {
                        C[l + j * *ldc] -= sum * A[l + i * *lda];
                    }
                }
            }
        } else {
            /* Q^T*C: apply reflections backward */
            for (i = *k - 1; i >= 0; i--) {
                if (tau[i] == 0.0f) continue;
                
                for (j = 0; j < *n; j++) {
                    sum = C[i + j * *ldc];
                    for (l = i + 1; l < *m; l++) {
                        sum += A[l + i * *lda] * C[l + j * *ldc];
                    }
                    sum *= tau[i];
                    
                    C[i + j * *ldc] -= sum;
                    for (l = i + 1; l < *m; l++) {
                        C[l + j * *ldc] -= sum * A[l + i * *lda];
                    }
                }
            }
        }
    } else {
        /* Apply Q or Q^T from right: C*op(Q) */
        if (is_notrans) {
            for (i = *k - 1; i >= 0; i--) {
                if (tau[i] == 0.0f) continue;
                
                for (j = 0; j < *m; j++) {
                    sum = C[j + i * *ldc];
                    for (l = i + 1; l < *n; l++) {
                        sum += C[j + l * *ldc] * A[l + i * *lda];
                    }
                    sum *= tau[i];
                    
                    C[j + i * *ldc] -= sum;
                    for (l = i + 1; l < *n; l++) {
                        C[j + l * *ldc] -= sum * A[l + i * *lda];
                    }
                }
            }
        } else {
            for (i = 0; i < *k; i++) {
                if (tau[i] == 0.0f) continue;
                
                for (j = 0; j < *m; j++) {
                    sum = C[j + i * *ldc];
                    for (l = i + 1; l < *n; l++) {
                        sum += C[j + l * *ldc] * A[l + i * *lda];
                    }
                    sum *= tau[i];
                    
                    C[j + i * *ldc] -= sum;
                    for (l = i + 1; l < *n; l++) {
                        C[j + l * *ldc] -= sum * A[l + i * *lda];
                    }
                }
            }
        }
    }
}
