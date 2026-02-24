/* sprod - Vector element-wise product */
#
void sprod_(int* n, float* x, int* incx, float* y, int* incy, float* prod) {
    if (*n <= 0) { *prod = 0.0f; return; }
    *prod = 0.0f;
    for (int i = 0; i < *n; i++) {
        *prod += x[i * (*incx)] * y[i * (*incy)];
    }
}
