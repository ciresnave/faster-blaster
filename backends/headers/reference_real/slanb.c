/* Band norm */
void slanb_(const char* norm, int* n, int* k, float* ab, int* ldab, float* work, float* result) {
    int n_val = *n;
    int k_val = *k;
    int ldab_val = *ldab;
    int i, j;
    float sum, max_val;
    
    /* Compute norm of banded matrix */
    max_val = 0.0f;
    
    if (norm[0] == 'F') {
        /* Frobenius norm */
        sum = 0.0f;
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < ldab_val; i++) {
                sum += ab[i + j*ldab_val] * ab[i + j*ldab_val];
            }
        }
        *result = (float)sqrt((double)sum);
    } else {
        /* Maximum absolute value */
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < ldab_val; i++) {
                float val = ab[i + j*ldab_val];
                if (val < 0.0f) val = -val;
                if (val > max_val) max_val = val;
            }
        }
        *result = max_val;
    }
}
