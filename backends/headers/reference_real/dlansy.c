#

/**
 * DLANSY - Compute norm of a symmetric matrix (double precision)
 *
 * Computes one of four matrix norms:
 * 'F' or 'E': Frobenius norm
 * '1': One norm (max column sum)
 * 'I': Infinity norm (max row sum)
 * 'M': Max norm (max absolute element)
 *
 * Parameters:
 * uplo - 'U' (upper triangular) or 'L' (lower triangular)
 * n - order of matrix
 * A - symmetric matrix
 * lda - leading dimension
 */
double dlansy_(const char *norm, const char *uplo, const int *n, const double *A,
               const int *lda, double *work)
{
    int i, j;
    double result = 0.0;
    double sum, value, scale;
    
    if (*n == 0) return 0.0;
    
    switch (norm[0]) {
        case 'M': /* Max norm */
            if (uplo[0] == 'U') {
                for (j = 0; j < *n; j++) {
                    for (i = 0; i <= j; i++) {
                        value = fabs(A[i + j * *lda]);
                        if (value > result) result = value;
                    }
                }
            } else {
                for (j = 0; j < *n; j++) {
                    for (i = j; i < *n; i++) {
                        value = fabs(A[i + j * *lda]);
                        if (value > result) result = value;
                    }
                }
            }
            break;
            
        case 'F': /* Frobenius norm */
        case 'E': {
            scale = 0.0;
            sum = 1.0;
            if (uplo[0] == 'U') {
                for (j = 0; j < *n; j++) {
                    for (i = 0; i < j; i++) {
                        value = fabs(A[i + j * *lda]);
                        if (value != 0.0) {
                            if (value > scale) {
                                sum = 1.0 + sum * (scale / value) * (scale / value);
                                scale = value;
                            } else {
                                sum = sum + (value / scale) * (value / scale);
                            }
                        }
                    }
                    value = fabs(A[j + j * *lda]);
                    if (value != 0.0) {
                        if (value > scale) {
                            sum = 1.0 + sum * (scale / value) * (scale / value);
                            scale = value;
                        } else {
                            sum = sum + (value / scale) * (value / scale);
                        }
                    }
                }
            } else {
                for (j = 0; j < *n; j++) {
                    value = fabs(A[j + j * *lda]);
                    if (value != 0.0) {
                        if (value > scale) {
                            sum = 1.0 + sum * (scale / value) * (scale / value);
                            scale = value;
                        } else {
                            sum = sum + (value / scale) * (value / scale);
                        }
                    }
                    for (i = j + 1; i < *n; i++) {
                        value = fabs(A[i + j * *lda]);
                        if (value != 0.0) {
                            if (value > scale) {
                                sum = 1.0 + sum * (scale / value) * (scale / value);
                                scale = value;
                            } else {
                                sum = sum + (value / scale) * (value / scale);
                            }
                        }
                    }
                }
            }
            result = scale * sqrt(sum);
            break;
        }
            
        case 'I': /* Infinity norm */
        case '1': /* One norm = Infinity norm for symmetric */
            if (uplo[0] == 'U') {
                for (i = 0; i < *n; i++) {
                    sum = fabs(A[i + i * *lda]);
                    for (j = i + 1; j < *n; j++) {
                        sum += fabs(A[i + j * *lda]);
                    }
                    for (j = 0; j < i; j++) {
                        sum += fabs(A[j + i * *lda]);
                    }
                    if (sum > result) result = sum;
                }
            } else {
                for (i = 0; i < *n; i++) {
                    sum = fabs(A[i + i * *lda]);
                    for (j = 0; j < i; j++) {
                        sum += fabs(A[i + j * *lda]);
                    }
                    for (j = i + 1; j < *n; j++) {
                        sum += fabs(A[j + i * *lda]);
                    }
                    if (sum > result) result = sum;
                }
            }
            break;
    }
    
    return result;
}
