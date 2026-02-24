/* spmd - Pseudo-minimal distance computation */
#
void spmd_(int* n, float* x, int* incx, float* y, int* incy, float* dist, int* info) {
    int n_val = *n;
    if (*n < 0) { *info = -1; return; }
    float sumsq = 0.0f;
    for (int i = 0; i < n_val; i++) {
        float diff = x[i * (*incx)] - y[i * (*incy)];
        sumsq += diff * diff;
    }
    *dist = sqrt(sumsq);
    *info = 0;
}
