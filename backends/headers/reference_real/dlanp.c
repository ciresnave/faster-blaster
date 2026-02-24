/* Packed norm */
void dlanp_(const char* norm, const char* uplo, int* n, double* ap, double* work, double* result) {
    int n_val = *n;
    int k = n_val * (n_val + 1) / 2;
    int i;
    double sum, max_val;
    
    max_val = 0.0;
    
    if (norm[0] == 'F') {
        /* Frobenius norm for packed matrix */
        sum = 0.0;
        for (i = 0; i < k; i++) {
            sum += ap[i] * ap[i];
        }
        *result = sqrt(sum);
    } else {
        /* Maximum absolute value */
        for (i = 0; i < k; i++) {
            double val = ap[i];
            if (val < 0.0) val = -val;
            if (val > max_val) max_val = val;
        }
        *result = max_val;
    }
}
