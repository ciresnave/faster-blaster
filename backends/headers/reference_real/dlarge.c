/* Eigenvalue perturb */
#
void dlarge_(int* n, double* a, int* lda, int* iseed, double* work, int* info) {
    int lda_val = *lda;
    if (*n < 0) { *info = -1; return; }
    if (*lda < *n) { *info = -3; return; }
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            a[i + j*lda_val] = (i == j) ? 1.0 : 0.0;
        }
    }
    *info = 0;
}
