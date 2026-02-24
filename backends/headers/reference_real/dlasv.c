/* SVD 2x2 */
void dlasv_(int* n, double* d, double* e, double* u, int* ldu, double* v, int* ldv, double* cs, double* sn, int* info) {
    int n_val = *n;
    int ldu_val = *ldu;
    int ldv_val = *ldv;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -1;
    } else if (ldu_val < n_val) {
        *info = -5;
    } else if (ldv_val < n_val) {
        *info = -7;
    }
    
    if (*info != 0) return;
    
    /* Initialize U and V as identity matrices */
    for (i = 0; i < n_val; i++) {
        u[i + i*ldu_val] = 1.0;
        v[i + i*ldv_val] = 1.0;
        cs[i] = 1.0;
        if (i < n_val - 1) sn[i] = 0.0;
    }
}
