/* Triangular H */
void dlarft_(const char* direct, const char* storev, int* n, int* k, double* v, int* ldv, double* tau, double* t, int* ldt, int* info) {
    int n_val = *n;
    int k_val = *k;
    int ldv_val = *ldv;
    int ldt_val = *ldt;
    int i, j;
    
    *info = 0;
    if (n_val < 0) {
        *info = -3;
    } else if (k_val < 0) {
        *info = -4;
    } else if (ldt_val < k_val) {
        *info = -8;
    } else if (ldv_val < n_val && storev[0] == 'C') {
        *info = -6;
    }
    
    if (*info != 0) return;
    
    /* Initialize T matrix to upper triangular form */
    for (j = 0; j < k_val; j++) {
        for (i = 0; i < k_val; i++) {
            t[i + j*ldt_val] = (i <= j) ? tau[j] : 0.0;
        }
    }
}
