/* dgrowf - Double precision growth factor */
#
void dgrowf_(int* n, double* x, int* incx, double* y, int* incy, double* growth) {
    if (*n <= 0) { *growth = 1.0; return; }
    double norm_x = 0.0, norm_y = 0.0;
    for (int i = 0; i < *n; i++) {
        double vx = fabs(x[i*(*incx)]);
        double vy = fabs(y[i*(*incy)]);
        if (vx > norm_x) norm_x = vx;
        if (vy > norm_y) norm_y = vy;
    }
    *growth = (norm_x > 0.0) ? norm_y / norm_x : 1.0;
}
