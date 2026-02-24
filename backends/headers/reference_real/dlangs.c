/* General norm */
void dlangs_(const char* norm, int* m, int* n, double* a, int* lda, double* work, double* result) {
    int m_val = *m;
    int n_val = *n;
    int lda_val = *lda;
    int i, j;
    double sum, max_val;
    
    max_val = 0.0;
    
    if (norm[0] == 'F') {
        /* Frobenius norm */
        sum = 0.0;
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < m_val; i++) {
                sum += a[i + j*lda_val] * a[i + j*lda_val];
            }
        }
        *result = sqrt(sum);
    } else {
        /* Maximum absolute value */
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < m_val; i++) {
                double val = a[i + j*lda_val];
                if (val < 0.0) val = -val;
                if (val > max_val) max_val = val;
            }
        }
        *result = max_val;
    }
}
