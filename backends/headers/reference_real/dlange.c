#

/**
 * DLANGE - Compute norm of a general matrix (double precision)
 *
 * Computes one of four matrix norms:
 * 'F' or 'E': Frobenius norm = sqrt(sum of squares of all elements)
 * '1': One norm = max column sum
 * 'I': Infinity norm = max row sum
 * 'M': Max norm = max absolute element
 */
double dlange_(const char *norm, const int *m, const int *n, const double *A,
               const int *lda, double *work)
{
    int i, j;
    double result = 0.0;
    double sum, value, scale;
    
    if (*m == 0 || *n == 0) return 0.0;
    
    switch (norm[0]) {
        case 'M': /* Max norm */
            for (j = 0; j < *n; j++) {
                for (i = 0; i < *m; i++) {
                    value = fabs(A[i + j * *lda]);
                    if (value > result) result = value;
                }
            }
            break;
            
        case 'F': /* Frobenius norm */
        case 'E': {
            scale = 0.0;
            sum = 1.0;
            for (j = 0; j < *n; j++) {
                for (i = 0; i < *m; i++) {
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
            result = scale * sqrt(sum);
            break;
        }
            
        case 'I': /* Infinity norm */
            for (i = 0; i < *m; i++) {
                sum = 0.0;
                for (j = 0; j < *n; j++) {
                    sum += fabs(A[i + j * *lda]);
                }
                if (sum > result) result = sum;
            }
            break;
            
        case '1': /* One norm */
            for (j = 0; j < *n; j++) {
                sum = 0.0;
                for (i = 0; i < *m; i++) {
                    sum += fabs(A[i + j * *lda]);
                }
                if (sum > result) result = sum;
            }
            break;
    }
    
    return result;
}
