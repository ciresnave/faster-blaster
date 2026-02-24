/* dlatdl - Double precision tridiagonal LU */
#
void dlatdl_(int* n, double* a, double* b, double* c, double* d, double* e, double* f, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
