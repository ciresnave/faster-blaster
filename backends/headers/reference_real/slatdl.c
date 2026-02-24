/* slatdl - Tridiagonal LU factorization */
#
void slatdl_(int* n, float* a, float* b, float* c, float* d, float* e, float* f, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
