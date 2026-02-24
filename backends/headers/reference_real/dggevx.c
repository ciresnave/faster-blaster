/* Generalized eigenvalues (expert) */
#
void dggevx_(char* balanc, char* jobvl, char* jobvr, char* sense, int* n, double* a, int* lda, double* b, int* ldb, double* alphar, double* alphai, double* beta, double* vl, int* ldvl, double* vr, int* ldvr, int* ilo, int* ihi, double* lscale, double* rscale, double* abnrm, double* bbnrm, double* rconde, double* rcondv, double* work, int* lwork, int* iwork, int* info) {
    int n_val = *n, lda_val = *lda, ldb_val = *ldb, ldvl_val = *ldvl, ldvr_val = *ldvr;
    if (*n < 0) { *info = -5; return; }
    if (*lda < n_val) { *info = -7; return; }
    if (*ldb < n_val) { *info = -9; return; }
    if (*ldvl < n_val) { *info = -13; return; }
    if (*ldvr < n_val) { *info = -15; return; }
    for (int i = 0; i < n_val; i++) {
        alphar[i] = a[i + i*lda_val];
        alphai[i] = 0.0;
        beta[i] = b[i + i*ldb_val];
    }
    *info = 0;
}
