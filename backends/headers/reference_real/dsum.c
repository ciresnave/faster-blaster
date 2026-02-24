/* dsum - Double precision vector sum */
#
void dsum_(int* n, double* x, int* incx, double* sum) {
    if (*n <= 0) { *sum = 0.0; return; }
    *sum = 0.0;
    for (int i = 0; i < *n; i++) {
        *sum += x[i * (*incx)];
    }
}
