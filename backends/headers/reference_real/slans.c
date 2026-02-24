/* Symmetric norm */
void slans_(const char* norm, const char* uplo, int* n, float* a, int* lda, float* work, float* result) {
    int n_val = *n;
    int lda_val = *lda;
    int i, j;
    float sum, max_val;
    
    max_val = 0.0f;
    
    if (norm[0] == 'F') {
        /* Frobenius norm */
        sum = 0.0f;
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < n_val; i++) {
                sum += a[i + j*lda_val] * a[i + j*lda_val];
            }
        }
        *result = (float)sqrt((double)sum);
    } else {
        /* Maximum absolute value */
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < n_val; i++) {
                float val = a[i + j*lda_val];
                if (val < 0.0f) val = -val;
                if (val > max_val) max_val = val;
            }
        }
        *result = max_val;
    }
}
