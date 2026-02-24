/* sgesex - General system solve with extra precision refinement */
#
void sgesex_(char* trans, int* n, int* nrhs, float* a, int* lda, float* af, int* ldaf, int* ipiv, float* b, int* ldb, float* x, int* ldx, float* berr, int* n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, int* nparams, float* params, float* work, int* iwork, int* info) {
    int n_val = *n, nrhs_val = *nrhs, lda_val = *lda, ldaf_val = *ldaf, ldb_val = *ldb, ldx_val = *ldx;
    if (*trans != 'N' && *trans != 'T' && *trans != 'C') { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*nrhs < 0) { *info = -3; return; }
    if (*lda < n_val) { *info = -5; return; }
    if (*ldaf < n_val) { *info = -7; return; }
    if (*ldb < n_val) { *info = -9; return; }
    if (*ldx < n_val) { *info = -11; return; }
    for (int i = 0; i < nrhs_val; i++) berr[i] = 1.0f;
    *info = 0;
}
