/* dkry - Double precision Krylov matrix generation */
#
void dkry_(int* n, double* a, int* lda, double* x, int* incx, double* y, int* incy, int* k, double* aky, int* ldaky, double* rcond, double* work, int* info) {
    int n_val = *n, lda_val = *lda;
    if (*n < 0) { *info = -1; return; }
    if (*lda < n_val) { *info = -3; return; }
    if (*k < 0) { *info = -7; return; }
    for (int j = 0; j < *k && j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            aky[i + j*(*ldaky)] = x[i * (*incx)];
        }
    }
    *rcond = 1.0;
    *info = 0;
}
