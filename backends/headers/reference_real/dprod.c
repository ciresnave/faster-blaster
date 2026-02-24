/* dprod - Double precision vector product */
#
void dprod_(int* n, double* x, int* incx, double* y, int* incy, double* prod) {
    if (*n <= 0) { *prod = 0.0; return; }
    *prod = 0.0;
    for (int i = 0; i < *n; i++) {
        *prod += x[i * (*incx)] * y[i * (*incy)];
    }
}
