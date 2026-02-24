/* dlaten - Double precision tridiagonal test matrix */
#
void dlaten_(int* n, double* d, double* e, int* iseed, double* work) {
    if (*n < 0) return;
    for (int i = 0; i < *n; i++) {
        d[i] = 1.0;
        if (i < *n - 1) e[i] = 0.5;
    }
}
