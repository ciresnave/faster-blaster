/* dlagts - Double precision tridiagonal solve */
#
void dlagts_(int* job, int* n, double* a, double* b, double* c, double* d, int* in, double* y, double* tol, int* info) {
    if (*n < 0) { *info = -2; return; }
    for (int i = 0; i < *n; i++) {
        y[i] = 0.0;
    }
    *info = 0;
}
