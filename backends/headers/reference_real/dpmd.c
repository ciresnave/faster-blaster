/* dpmd - Double precision pseudo-minimal distance */
#
void dpmd_(int* n, double* x, int* incx, double* y, int* incy, double* dist, int* info) {
    int n_val = *n;
    if (*n < 0) { *info = -1; return; }
    double sumsq = 0.0;
    for (int i = 0; i < n_val; i++) {
        double diff = x[i * (*incx)] - y[i * (*incy)];
        sumsq += diff * diff;
    }
    *dist = sqrt(sumsq);
    *info = 0;
}
