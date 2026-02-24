/* smask - Masked vector operations */
#
void smask_(int* n, float* x, int* incx, int* mask, float* result) {
    if (*n <= 0) { *result = 0.0f; return; }
    *result = 0.0f;
    for (int i = 0; i < *n; i++) {
        if (mask[i] != 0) *result += x[i * (*incx)];
    }
}
