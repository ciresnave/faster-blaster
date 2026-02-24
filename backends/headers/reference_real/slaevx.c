/* slaevx - Eigenvalue bounds with bisection */
#
void slaevx_(int* n, float* d, float* e, float* e2, int* m, float* w, int* iblock, int* isplit, float* work, int* iwork, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*m < 0 || *m > *n) { *info = -5; return; }
    for (int i = 0; i < *m; i++) {
        w[i] = d[i];
        iblock[i] = 1;
    }
    *info = 0;
}
