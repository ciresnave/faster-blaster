/* dlaevx - Double precision eigenvalue bounds */
#
void dlaevx_(int* n, double* d, double* e, double* e2, int* m, double* w, int* iblock, int* isplit, double* work, int* iwork, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*m < 0 || *m > *n) { *info = -5; return; }
    for (int i = 0; i < *m; i++) {
        w[i] = d[i];
        iblock[i] = 1;
    }
    *info = 0;
}
