/* dmax - Double precision maximum with index */
#
void dmax_(int* n, double* x, int* incx, int* idx) {
    if (*n <= 0) { *idx = 0; return; }
    double maxval = fabs(x[0]);
    *idx = 1;
    for (int i = 1; i < *n; i++) {
        double absval = fabs(x[i * (*incx)]);
        if (absval > maxval) { maxval = absval; *idx = i + 1; }
    }
}
