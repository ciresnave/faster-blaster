/* Eigenvalue perturb */
#
void slarge_(int* n, float* a, int* lda, int* iseed, float* work, int* info) {
    int lda_val = *lda;
    if (*n < 0) { *info = -1; return; }
    if (*lda < *n) { *info = -3; return; }
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            a[i + j*lda_val] = (i == j) ? 1.0f : 0.0f;
        }
    }
    *info = 0;
}
