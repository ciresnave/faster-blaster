/* dmask - Double precision masked operations */
#
void dmask_(int* n, double* x, int* incx, int* mask, double* result) {
    if (*n <= 0) { *result = 0.0; return; }
    *result = 0.0;
    for (int i = 0; i < *n; i++) {
        if (mask[i] != 0) *result += x[i * (*incx)];
    }
}
