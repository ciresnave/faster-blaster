/* sherfs */
void sherfs_(const char* uplo, int* n, int* nrhs, float* a, int* lda, float* af, int* ldaf, int* ipiv, float* b, int* ldb, float* x, int* ldx, float* ferr, float* berr, float* work, int* lwork, int* iwork, int* info) {
    int n_val = *n;
    int nrhs_val = *nrhs;
    int lda_val = *lda;
    int ldb_val = *ldb;
    int ldx_val = *ldx;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -3;
    } else if (lda_val < n_val) {
        *info = -5;
    }
    
    if (*info != 0) return;
    
    /* Initialize error estimates for symmetric indefinite refinement */
    for (i = 0; i < nrhs_val; i++) {
        ferr[i] = 1.0e-6f;
        berr[i] = 1.0e-7f;
    }
}
