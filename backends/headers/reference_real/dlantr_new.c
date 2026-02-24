#

/**
 * DLANTR - Compute norm of a triangular matrix (double precision)
 */
double dlantr_(const char *norm, const char *uplo, const char *diag, const int *m,
               const int *n, const double *A, const int *lda, double *work)
{
    int i, j;
    double result = 0.0, sum, value;
    int diag_is_unit = (diag[0] == 'U' || diag[0] == 'u');
    
    if (*m == 0 || *n == 0) return 0.0;
    
    if (norm[0] == 'M') {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < *n; j++) {
                for (i = 0; i <= (j < *m ? j : *m-1); i++) {
                    value = fabs(A[i + j * *lda]);
                    if (value > result) result = value;
                }
            }
        } else {
            for (j = 0; j < *n; j++) {
                for (i = j; i < *m; i++) {
                    value = fabs(A[i + j * *lda]);
                    if (value > result) result = value;
                }
            }
        }
        if (diag_is_unit) result = 1.0;
    } else if (norm[0] == 'I') {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (i = 0; i < *m; i++) {
                sum = diag_is_unit ? 1.0 : (i < *n ? fabs(A[i + i * *lda]) : 0.0);
                for (j = i + 1; j < *n; j++) {
                    sum += fabs(A[i + j * *lda]);
                }
                if (sum > result) result = sum;
            }
        } else {
            for (i = 0; i < *m; i++) {
                sum = 0.0;
                for (j = 0; j < i && j < *n; j++) {
                    sum += fabs(A[i + j * *lda]);
                }
                if (i < *n) sum += diag_is_unit ? 1.0 : fabs(A[i + i * *lda]);
                if (sum > result) result = sum;
            }
        }
    } else if (norm[0] == '1') {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < *n; j++) {
                sum = 0.0;
                for (i = 0; i <= (j < *m ? j : *m-1); i++) {
                    sum += fabs(A[i + j * *lda]);
                }
                if (sum > result) result = sum;
            }
        } else {
            for (j = 0; j < *n; j++) {
                sum = 0.0;
                for (i = j; i < *m; i++) {
                    sum += fabs(A[i + j * *lda]);
                }
                if (sum > result) result = sum;
            }
        }
    } else { /* Frobenius */
        sum = 0.0;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < *n; j++) {
                for (i = 0; i <= (j < *m ? j : *m-1); i++) {
                    value = fabs(A[i + j * *lda]);
                    sum += value * value;
                }
            }
        } else {
            for (j = 0; j < *n; j++) {
                for (i = j; i < *m; i++) {
                    value = fabs(A[i + j * *lda]);
                    sum += value * value;
                }
            }
        }
        if (diag_is_unit) sum += (double)(*m < *n ? *m : *n);
        result = sqrt(sum);
    }
    
    return result;
}
