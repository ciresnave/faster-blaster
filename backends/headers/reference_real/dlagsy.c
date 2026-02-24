/* dlagsy - Double precision symmetric random test matrix */
#
void dlagsy_(int* n, int* k, double* d, double* a, int* lda, int* iseed, double* work, int* info) {
    int n_val = *n, lda_val = *lda;
    if (*n < 0) { *info = -1; return; }
    if (*k < 0 || *k > *n - 1) { *info = -2; return; }
    if (*lda < n_val) { *info = -5; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            a[i + j*lda_val] = 0.0;
        }
        a[j + j*lda_val] = d[j];
    }
    *info = 0;
}
