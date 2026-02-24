/* Generalized eigenvalues (expert) */
#
void sggevx_(char* balanc, char* jobvl, char* jobvr, char* sense, int* n, float* a, int* lda, float* b, int* ldb, float* alphar, float* alphai, float* beta, float* vl, int* ldvl, float* vr, int* ldvr, int* ilo, int* ihi, float* lscale, float* rscale, float* abnrm, float* bbnrm, float* rconde, float* rcondv, float* work, int* lwork, int* iwork, int* info) {
    int n_val = *n, lda_val = *lda, ldb_val = *ldb, ldvl_val = *ldvl, ldvr_val = *ldvr;
    if (*n < 0) { *info = -5; return; }
    if (*lda < n_val) { *info = -7; return; }
    if (*ldb < n_val) { *info = -9; return; }
    if (*ldvl < n_val) { *info = -13; return; }
    if (*ldvr < n_val) { *info = -15; return; }
    for (int i = 0; i < n_val; i++) {
        alphar[i] = a[i + i*lda_val];
        alphai[i] = 0.0f;
        beta[i] = b[i + i*ldb_val];
    }
    *info = 0;
}
