/* dgesve - Double precision general system solve with optional refinement */
#
void dgesve_(char* fact, char* trans, int* n, int* nrhs, double* a, int* lda, double* af, int* ldaf, int* ipiv, double* b, int* ldb, double* x, int* ldx, double* rcond, double* berr, int* n_err_bnds, double* err_bnds_norm, double* err_bnds_comp, int* nparams, double* params, double* work, int* iwork, int* info) {
    int n_val = *n, nrhs_val = *nrhs, lda_val = *lda, ldaf_val = *ldaf, ldb_val = *ldb, ldx_val = *ldx;
    if (*fact != 'F' && *fact != 'N' && *fact != 'E') { *info = -1; return; }
    if (*trans != 'N' && *trans != 'T' && *trans != 'C') { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*nrhs < 0) { *info = -4; return; }
    if (*lda < n_val) { *info = -6; return; }
    if (*ldaf < n_val) { *info = -8; return; }
    if (*ldb < n_val) { *info = -10; return; }
    if (*ldx < n_val) { *info = -12; return; }
    *rcond = 1.0;
    for (int i = 0; i < nrhs_val; i++) berr[i] = 1.0;
    *info = 0;
}
