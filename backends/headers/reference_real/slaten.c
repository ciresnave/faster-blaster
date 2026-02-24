/* slaten - Tridiagonal test matrix */
#
void slaten_(int* n, float* d, float* e, int* iseed, float* work) {
    if (*n < 0) return;
    for (int i = 0; i < *n; i++) {
        d[i] = 1.0f;
        if (i < *n - 1) e[i] = 0.5f;
    }
}
