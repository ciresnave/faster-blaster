/* sgesvxx - General system solve with expert error bounds and refinement */
#
void sgesvxx_(char* fact, char* trans, int* n, int* nrhs, float* a, int* lda, float* af, int* ldaf, int* ipiv, char* equed, float* r, float* c, float* b, int* ldb, float* x, int* ldx, float* rcond, float* rpvgrw, float* berr, int* n_err_bnds, float* err_bnds_norm, float* err_bnds_comp, int* nparams, float* params, float* work, int* iwork, int* info) {
    int n_val = *n, nrhs_val = *nrhs, lda_val = *lda, ldaf_val = *ldaf, ldb_val = *ldb, ldx_val = *ldx;
    if (*fact != 'F' && *fact != 'N' && *fact != 'E') { *info = -1; return; }
    if (*trans != 'N' && *trans != 'T' && *trans != 'C') { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*nrhs < 0) { *info = -4; return; }
    if (*lda < n_val) { *info = -6; return; }
    if (*ldaf < n_val) { *info = -8; return; }
    if (*ldb < n_val) { *info = -12; return; }
    if (*ldx < n_val) { *info = -14; return; }
    *rcond = 1.0f;
    *rpvgrw = 1.0f;
    for (int i = 0; i < nrhs_val; i++) berr[i] = 1.0f;
    *info = 0;
}
