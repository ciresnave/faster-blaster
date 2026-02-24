#

/**
 * SLANGE - Compute norm of a general matrix (single precision)
 *
 * Computes one of four matrix norms:
 * 'F' or 'E': Frobenius norm = sqrt(sum of squares of all elements)
 * '1': One norm = max column sum
 * 'I': Infinity norm = max row sum
 * 'M': Max norm = max absolute element
 */
float slange_(const char *norm, const int *m, const int *n, const float *A,
              const int *lda, float *work)
{
    int i, j;
    float result = 0.0f;
    float sum, value, scale;
    
    if (*m == 0 || *n == 0) return 0.0f;
    
    switch (norm[0]) {
        case 'M': /* Max norm */
            for (j = 0; j < *n; j++) {
                for (i = 0; i < *m; i++) {
                    value = fabsf(A[i + j * *lda]);
                    if (value > result) result = value;
                }
            }
            break;
            
        case 'F': /* Frobenius norm */
        case 'E': {
            scale = 0.0f;
            sum = 1.0f;
            for (j = 0; j < *n; j++) {
                for (i = 0; i < *m; i++) {
                    value = fabsf(A[i + j * *lda]);
                    if (value != 0.0f) {
                        if (value > scale) {
                            sum = 1.0f + sum * (scale / value) * (scale / value);
                            scale = value;
                        } else {
                            sum = sum + (value / scale) * (value / scale);
                        }
                    }
                }
            }
            result = scale * sqrtf(sum);
            break;
        }
            
        case 'I': /* Infinity norm */
            for (i = 0; i < *m; i++) {
                sum = 0.0f;
                for (j = 0; j < *n; j++) {
                    sum += fabsf(A[i + j * *lda]);
                }
                if (sum > result) result = sum;
            }
            break;
            
        case '1': /* One norm */
            for (j = 0; j < *n; j++) {
                sum = 0.0f;
                for (i = 0; i < *m; i++) {
                    sum += fabsf(A[i + j * *lda]);
                }
                if (sum > result) result = sum;
            }
            break;
    }
    
    return result;
}
