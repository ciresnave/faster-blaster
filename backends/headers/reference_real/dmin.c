/* dmin - Double precision minimum with index */
#
void dmin_(int* n, double* x, int* incx, int* idx) {
    if (*n <= 0) { *idx = 0; return; }
    double minval = fabs(x[0]);
    *idx = 1;
    for (int i = 1; i < *n; i++) {
        double absval = fabs(x[i * (*incx)]);
        if (absval < minval) { minval = absval; *idx = i + 1; }
    }
}
