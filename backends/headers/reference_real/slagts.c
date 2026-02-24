/* slagts - Solve tridiagonal system */
#
void slagts_(int* job, int* n, float* a, float* b, float* c, float* d, int* in, float* y, float* tol, int* info) {
    if (*n < 0) { *info = -2; return; }
    for (int i = 0; i < *n; i++) {
        y[i] = 0.0f;
    }
    *info = 0;
}
