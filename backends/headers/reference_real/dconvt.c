/* dconvt - Double precision convolution */
#
void dconvt_(int* n, double* x, int* incx, double* h, int* inch, double* y, int* incy, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        double sum = 0.0;
        for (int j = 0; j < *n - i; j++) {
            sum += x[j * (*incx)] * h[(i + j) * (*inch)];
        }
        y[i * (*incy)] = sum;
    }
    *info = 0;
}
