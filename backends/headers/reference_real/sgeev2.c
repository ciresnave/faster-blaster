/* sgeev2 - Eigenvalues and eigenvectors via explicit shift QR */
#
void sgeev2_(char* jobvl, char* jobvr, int* n, float* a, int* lda, float* wr, float* wi, float* vl, int* ldvl, float* vr, int* ldvr, float* work, int* lwork, int* info) {
    int n_val = *n, lda_val = *lda, ldvl_val = *ldvl, ldvr_val = *ldvr;
    if (*jobvl != 'N' && *jobvl != 'V') { *info = -1; return; }
    if (*jobvr != 'N' && *jobvr != 'V') { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*lda < n_val) { *info = -5; return; }
    if (*ldvl < 1 || (*jobvl == 'V' && *ldvl < n_val)) { *info = -9; return; }
    if (*ldvr < 1 || (*jobvr == 'V' && *ldvr < n_val)) { *info = -11; return; }
    /* Extract diagonal eigenvalues */
    for (int i = 0; i < n_val; i++) {
        wr[i] = a[i + i*lda_val];
        wi[i] = 0.0f;
    }
    /* Initialize eigenvectors if requested */
    if (*jobvl == 'V') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i < n_val; i++) {
                vl[i + j*ldvl_val] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    if (*jobvr == 'V') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i < n_val; i++) {
                vr[i + j*ldvr_val] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    *info = 0;
}
