#

/**
 * DLANBS - Compute norm of a symmetric banded matrix (double precision)
 * 
 * Computes norm of an n-by-n symmetric band matrix A with
 * ku superdiagonals and kl subdiagonals (stored in compact band form).
 * 
 * Norms computed:
 *  'M' - max(abs(A))
 *  'F' - Frobenius norm (sqrt of sum of squares)
 *  '1' - one-norm (max column sum)
 *  'I' - infinity norm (max row sum, same as '1' for symmetric)
 */
double dlanbs_(const char *norm, const char *uplo, const int *n, const int *k,
               const double *AB, const int *ldab, double *work)
{
    int i, j;
    double result = 0.0, sum, value;
    int k_band = *k;
    
    if (*n == 0) return 0.0;
    
    if (norm[0] == 'M') {
        /* Max absolute value */
        for (j = 0; j < *n; j++) {
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                /* Upper triangular band: columns k to n with offset */
                int istart = k_band > j ? k_band - j : 0;
                for (i = istart; i <= k_band; i++) {
                    value = fabs(AB[i + j * *ldab]);
                    if (value > result) result = value;
                }
            } else {
                /* Lower triangular band */
                for (i = 0; i <= k_band && j + i < *n; i++) {
                    value = fabs(AB[i + j * *ldab]);
                    if (value > result) result = value;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'E' || norm[0] == 'e') {
        /* Frobenius norm with scaling for stability */
        double scale = 0.0, ssq = 1.0;
        for (j = 0; j < *n; j++) {
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                int istart = k_band > j ? k_band - j : 0;
                for (i = istart; i <= k_band; i++) {
                    value = fabs(AB[i + j * *ldab]);
                    if (value != 0.0) {
                        if (scale < value) {
                            ssq = 1.0 + ssq * (scale / value) * (scale / value);
                            scale = value;
                        } else {
                            ssq += (value / scale) * (value / scale);
                        }
                    }
                }
            } else {
                for (i = 0; i <= k_band && j + i < *n; i++) {
                    value = fabs(AB[i + j * *ldab]);
                    if (value != 0.0) {
                        if (scale < value) {
                            ssq = 1.0 + ssq * (scale / value) * (scale / value);
                            scale = value;
                        } else {
                            ssq += (value / scale) * (value / scale);
                        }
                    }
                }
            }
        }
        result = scale * sqrt(ssq);
    } else if (norm[0] == '1' || norm[0] == 'I' || norm[0] == 'i') {
        /* One/Infinity norm (same for symmetric matrices) */
        for (i = 0; i < *n; i++) {
            sum = 0.0;
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                /* Column j, row i: stored in position [k_band - j + i, j] */
                if (i <= k_band) {
                    /* Elements to the right */
                    for (j = i; j < *n && j - i <= k_band; j++) {
                        sum += fabs(AB[k_band - j + i + j * *ldab]);
                    }
                }
            } else {
                /* Row i contributions */
                for (j = i - k_band > 0 ? i - k_band : 0; j < i; j++) {
                    sum += fabs(AB[i - j + j * *ldab]);
                }
                sum += fabs(AB[0 + i * *ldab]); /* Diagonal */
                for (j = i + 1; j < *n && j - i <= k_band; j++) {
                    sum += fabs(AB[j - i + i * *ldab]);
                }
            }
            if (sum > result) result = sum;
        }
    }
    
    return result;
}
