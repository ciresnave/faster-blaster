/* ssum - Vector element sum */
#
void ssum_(int* n, float* x, int* incx, float* sum) {
    if (*n <= 0) { *sum = 0.0f; return; }
    *sum = 0.0f;
    for (int i = 0; i < *n; i++) {
        *sum += x[i * (*incx)];
    }
}
