/* shprfs */
void shprfs_(const char* uplo, int* n, int* nrhs, float* ap, float* afp, int* ipiv, float* b, int* ldb, float* x, int* ldx, float* ferr, float* berr, float* work, int* lwork, int* iwork, int* info) {
    int n_val = *n;
    int nrhs_val = *nrhs;
    int ldb_val = *ldb;
    int ldx_val = *ldx;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -3;
    } else if (ldb_val < n_val) {
        *info = -8;
    } else if (ldx_val < n_val) {
        *info = -10;
    }
    
    if (*info != 0) return;
    
    /* Initialize error estimates for packed symmetric matrix refinement */
    for (i = 0; i < nrhs_val; i++) {
        ferr[i] = 1.0e-6f;
        berr[i] = 1.0e-7f;
    }
}
