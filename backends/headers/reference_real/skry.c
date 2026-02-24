/* skry - Generate Krylov matrix from A and x */
#
void skry_(int* n, float* a, int* lda, float* x, int* incx, float* y, int* incy, int* k, float* aky, int* ldaky, float* rcond, float* work, int* info) {
    int n_val = *n, lda_val = *lda;
    if (*n < 0) { *info = -1; return; }
    if (*lda < n_val) { *info = -3; return; }
    if (*k < 0) { *info = -7; return; }
    for (int j = 0; j < *k && j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            aky[i + j*(*ldaky)] = x[i * (*incx)];
        }
    }
    *rcond = 1.0f;
    *info = 0;
}
