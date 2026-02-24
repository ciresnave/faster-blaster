/* Apply H with Y */
void dlarfy_(const char* uplo, int* n, double* v, int* incv, double* tau, double* c, int* ldc, double* work, int* info) {
    int n_val = *n;
    int incv_val = *incv;
    int ldc_val = *ldc;
    double tau_val = *tau;
    int i, j;
    
    *info = 0;
    if (ldc_val < n_val) {
        *info = -7;
    }
    
    if (*info != 0) return;
    
    /* Apply symmetric Householder reduction: C := (I - 2*tau*v*v^T)*C*(I - 2*tau*v*v^T)^T */
    if (tau_val != 0.0) {
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < n_val; i++) {
                c[i + j*ldc_val] = c[i + j*ldc_val] * (1.0 - 2.0 * tau_val * v[i*incv_val] * v[j*incv_val]);
            }
        }
    }
}
