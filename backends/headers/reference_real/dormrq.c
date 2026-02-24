#

void dormrq_(const char *side, const char *trans, const int *m, const int *n,
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
                int row = *m - *k + i;
                for (j = 0; j < *n; j++) {
                    sum = C[row + j * *ldc];
                    for (l = 0; l <= row; l++) {
                        sum += A[row + l * *lda] * C[l + j * *ldc];
                    }
                    sum *= tau[i];
                    C[row + j * *ldc] -= sum;
                    for (l = 0; l <= row; l++) {
                        C[l + j * *ldc] -= sum * A[row + l * *lda];
                    }
                }
            }
        } else {
            for (i = 0; i < *k; i++) {
                if (tau[i] == 0.0) continue;
                int row = *m - *k + i;
                for (j = 0; j < *n; j++) {
                    sum = C[row + j * *ldc];
                    for (l = 0; l <= row; l++) {
                        sum += A[row + l * *lda] * C[l + j * *ldc];
                    }
                    sum *= tau[i];
                    C[row + j * *ldc] -= sum;
                    for (l = 0; l <= row; l++) {
                        C[l + j * *ldc] -= sum * A[row + l * *lda];
                    }
                }
            }
        }
    } else {
        if (is_notrans) {
            for (i = 0; i < *k; i++) {
                if (tau[i] == 0.0) continue;
                int col = *n - *k + i;
                for (j = 0; j < *m; j++) {
                    sum = C[j + col * *ldc];
                    for (l = 0; l <= col; l++) {
                        sum += C[j + l * *ldc] * A[col + l * *lda];
                    }
                    sum *= tau[i];
                    C[j + col * *ldc] -= sum;
                    for (l = 0; l <= col; l++) {
                        C[j + l * *ldc] -= sum * A[col + l * *lda];
                    }
                }
            }
        } else {
            for (i = *k - 1; i >= 0; i--) {
                if (tau[i] == 0.0) continue;
                int col = *n - *k + i;
                for (j = 0; j < *m; j++) {
                    sum = C[j + col * *ldc];
                    for (l = 0; l <= col; l++) {
                        sum += C[j + l * *ldc] * A[col + l * *lda];
                    }
                    sum *= tau[i];
                    C[j + col * *ldc] -= sum;
                    for (l = 0; l <= col; l++) {
                        C[j + l * *ldc] -= sum * A[col + l * *lda];
                    }
                }
            }
        }
    }
}
