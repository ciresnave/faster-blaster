/* dgbrfs */
void dgbrfs_(const char* trans, int* n, int* kl, int* ku, int* nrhs, double* ab, int* ldab, double* afb, int* ldafb, int* ipiv, double* b, int* ldb, double* x, int* ldx, double* ferr, double* berr, double* work, int* lwork, int* iwork, int* info) {
    int n_val = *n;
    int nrhs_val = *nrhs;
    int ldb_val = *ldb;
    int ldx_val = *ldx;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -2;
    } else if (ldb_val < n_val) {
        *info = -10;
    } else if (ldx_val < n_val) {
        *info = -12;
    }
    
    if (*info != 0) return;
    
    /* Initialize error estimates for banded matrix refinement */
    for (i = 0; i < nrhs_val; i++) {
        ferr[i] = 1.0e-15;
        berr[i] = 1.0e-16;
    }
}
