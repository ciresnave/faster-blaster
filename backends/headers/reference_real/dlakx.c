/* dlakx - Double precision eigenvalue bounds with shift */
#
void dlakx_(int* n, double* d, double* e, int* k, double* c, double* s, double* xmin, double* xmax) {
    if (*n < 1) { *xmin = 0.0; *xmax = 0.0; return; }
    double radius = fabs(*c) + fabs(*s);
    *xmin = *c - radius;
    *xmax = *c + radius;
}
