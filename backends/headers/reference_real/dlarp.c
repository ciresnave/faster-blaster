/* dlarp - Double precision Givens rotation */
#
void dlarp_(int* j, double* v, int* incv, double* x, int* incx, double* c, double* s) {
    if (*j <= 0) return;
    double x0 = x[0];
    double v0 = v[0];
    x[0] = *c * x0 + *s * v0;
    v[0] = -(*s) * x0 + *c * v0;
}
