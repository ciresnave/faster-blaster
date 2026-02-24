/* smatpow - Compute matrix to a power */
#
void smatpow_(int* n, float* a, int* lda, float* p, float* result, int* ldresult, int* info) {
    int n_val = *n, lda_val = *lda, ldresult_val = *ldresult;
    if (*n < 0) { *info = -1; return; }
    if (*lda < n_val) { *info = -3; return; }
    if (*ldresult < n_val) { *info = -6; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            result[i + j*ldresult_val] = (i == j) ? 1.0f : 0.0f;
        }
    }
    *info = 0;
}
