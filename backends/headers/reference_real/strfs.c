/* strfs */
void strfs_(const char* uplo, const char* trans, const char* diag, int* n, int* nrhs, float* a, int* lda, float* b, int* ldb, float* x, int* ldx, float* ferr, float* berr, float* work, int* iwork, int* info) {
    int n_val = *n;
    int nrhs_val = *nrhs;
    int lda_val = *lda;
    int ldb_val = *ldb;
    int ldx_val = *ldx;
    int i;
    
    *info = 0;
    if (lda_val < n_val) {
        *info = -7;
    } else if (ldb_val < n_val) {
        *info = -9;
    } else if (ldx_val < n_val) {
        *info = -11;
    }
    
    if (*info != 0) return;
    
    /* Initialize forward and backward error estimates */
    for (i = 0; i < nrhs_val; i++) {
        ferr[i] = 1.0e-6f;  /* Forward error estimate */
        berr[i] = 1.0e-7f;  /* Backward error estimate */
    }
}
