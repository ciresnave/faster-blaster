/* Packed norm */
void slanp_(const char* norm, const char* uplo, int* n, float* ap, float* work, float* result) {
    int n_val = *n;
    int k = n_val * (n_val + 1) / 2;
    int i;
    float sum, max_val;
    
    max_val = 0.0f;
    
    if (norm[0] == 'F') {
        /* Frobenius norm for packed matrix */
        sum = 0.0f;
        for (i = 0; i < k; i++) {
            sum += ap[i] * ap[i];
        }
        *result = (float)sqrt((double)sum);
    } else {
        /* Maximum absolute value */
        for (i = 0; i < k; i++) {
            float val = ap[i];
            if (val < 0.0f) val = -val;
            if (val > max_val) max_val = val;
        }
        *result = max_val;
    }
}
