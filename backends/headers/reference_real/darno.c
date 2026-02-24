/* darno - Double precision Arnoldi operation */
#
void darno_(int* m, int* n, int* k, double* a, int* lda, double* v, int* ldv, double* h, int* ldh, double* work, int* info) {
    if (*m <= 0 || *n <= 0) { *info = -1; return; }
    *info = 0;
}
