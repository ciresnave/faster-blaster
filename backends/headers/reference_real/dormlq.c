#

void dormlq_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const double *A, const int *lda, const double *tau,
             double *C, const int *ldc, double *work, const int *lwork, int *info)
{
    int i, j, l;
    double sum;
    int is_left = (side[0] == 'L' || side[0] == 'l');
    int is_notrans = (trans[0] == 'N' || trans[0] == 'n');
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    if (is_left) {
        if (is_notrans) {
            for (i = *k - 1; i >= 0; i--) {
                if (tau[i] == 0.0) continue;
                for (j = 0; j < *n; j++) {
                    sum = C[i + j * *ldc];
                    for (l = i + 1; l < *m; l++) {
                        sum += A[i + l * *lda] * C[l + j * *ldc];
                    }
                    sum *= tau[i];
                    C[i + j * *ldc] -= sum;
                    for (l = i + 1; l < *m; l++) {
                        C[l + j * *ldc] -= sum * A[i + l * *lda];
                    }
                }
            }
        } else {
            for (i = 0; i < *k; i++) {
                if (tau[i] == 0.0) continue;
                for (j = 0; j < *n; j++) {
                    sum = C[i + j * *ldc];
                    for (l = i + 1; l < *m; l++) {
                        sum += A[i + l * *lda] * C[l + j * *ldc];
                    }
                    sum *= tau[i];
                    C[i + j * *ldc] -= sum;
                    for (l = i + 1; l < *m; l++) {
                        C[l + j * *ldc] -= sum * A[i + l * *lda];
                    }
                }
            }
        }
    } else {
        if (is_notrans) {
            for (i = 0; i < *k; i++) {
                if (tau[i] == 0.0) continue;
                for (j = 0; j < *m; j++) {
                    sum = C[j + i * *ldc];
                    for (l = i + 1; l < *n; l++) {
                        sum += C[j + l * *ldc] * A[i + l * *lda];
                    }
                    sum *= tau[i];
                    C[j + i * *ldc] -= sum;
                    for (l = i + 1; l < *n; l++) {
                        C[j + l * *ldc] -= sum * A[i + l * *lda];
                    }
                }
            }
        } else {
            for (i = *k - 1; i >= 0; i--) {
                if (tau[i] == 0.0) continue;
                for (j = 0; j < *m; j++) {
                    sum = C[j + i * *ldc];
                    for (l = i + 1; l < *n; l++) {
                        sum += C[j + l * *ldc] * A[i + l * *lda];
                    }
                    sum *= tau[i];
                    C[j + i * *ldc] -= sum;
                    for (l = i + 1; l < *n; l++) {
                        C[j + l * *ldc] -= sum * A[i + l * *lda];
                    }
                }
            }
        }
    }
}
