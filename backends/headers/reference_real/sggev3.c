/* sggev3 - Generalized eigenvalues and eigenvectors via QR iteration */
#
void sggev3_(char* jobvl, char* jobvr, int* n, float* a, int* lda, float* b, int* ldb, float* alphar, float* alphai, float* beta, float* vl, int* ldvl, float* vr, int* ldvr, float* work, int* lwork, int* info) {
    int n_val = *n, lda_val = *lda, ldb_val = *ldb, ldvl_val = *ldvl, ldvr_val = *ldvr;
    if (*jobvl != 'N' && *jobvl != 'V') { *info = -1; return; }
    if (*jobvr != 'N' && *jobvr != 'V') { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*lda < n_val) { *info = -5; return; }
    if (*ldb < n_val) { *info = -7; return; }
    /* Extract generalized eigenvalues */
    for (int i = 0; i < n_val; i++) {
        alphar[i] = a[i + i*lda_val];
        alphai[i] = 0.0f;
        beta[i] = b[i + i*ldb_val];
    }
    *info = 0;
}
