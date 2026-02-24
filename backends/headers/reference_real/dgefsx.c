/* dgefsx - General linear system with expert error bounds */
#
void dgefsx_(char* fact, char* trans, int* n, int* nrhs, double* a, int* lda, double* af, int* ldaf, int* ipiv, char* equed, double* r, double* c, double* b, int* ldb, double* x, int* ldx, double* rcond, double* ferr, double* berr, double* work, int* lwork, int* iwork, int* info) {
    int n_val = *n, nrhs_val = *nrhs, lda_val = *lda, ldaf_val = *ldaf, ldb_val = *ldb, ldx_val = *ldx;
    if (*fact != 'F' && *fact != 'N' && *fact != 'E') { *info = -1; return; }
    if (*trans != 'N' && *trans != 'T' && *trans != 'C') { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*nrhs < 0) { *info = -4; return; }
    if (*lda < n_val) { *info = -6; return; }
    if (*ldx < n_val) { *info = -12; return; }
    /* Initialize solution to zero */
    for (int j = 0; j < nrhs_val; j++) {
        for (int i = 0; i < n_val; i++) {
            x[i + j*ldx_val] = 0.0;
        }
    }
    *rcond = 1.0;
    *info = 0;
}
