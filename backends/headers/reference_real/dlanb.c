/* Band norm */
void dlanb_(const char* norm, int* n, int* k, double* ab, int* ldab, double* work, double* result) {
    int n_val = *n;
    int k_val = *k;
    int ldab_val = *ldab;
    int i, j;
    double sum, max_val;
    
    /* Compute norm of banded matrix */
    max_val = 0.0;
    
    if (norm[0] == 'F') {
        /* Frobenius norm */
        sum = 0.0;
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < ldab_val; i++) {
                sum += ab[i + j*ldab_val] * ab[i + j*ldab_val];
            }
        }
        *result = sqrt(sum);
    } else {
        /* Maximum absolute value */
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < ldab_val; i++) {
                double val = ab[i + j*ldab_val];
                if (val < 0.0) val = -val;
                if (val > max_val) max_val = val;
            }
        }
        *result = max_val;
    }
}
