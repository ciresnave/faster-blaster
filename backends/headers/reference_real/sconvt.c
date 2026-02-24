/* sconvt - Convolution computation */
#
void sconvt_(int* n, float* x, int* incx, float* h, int* inch, float* y, int* incy, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < *n - i; j++) {
            sum += x[j * (*incx)] * h[(i + j) * (*inch)];
        }
        y[i * (*incy)] = sum;
    }
    *info = 0;
}
