/* slakx - Eigenvalue bounds with spectral shift */
#
void slakx_(int* n, float* d, float* e, int* k, float* c, float* s, float* xmin, float* xmax) {
    if (*n < 1) { *xmin = 0.0f; *xmax = 0.0f; return; }
    float radius = fabs(*c) + fabs(*s);
    *xmin = *c - radius;
    *xmax = *c + radius;
}
