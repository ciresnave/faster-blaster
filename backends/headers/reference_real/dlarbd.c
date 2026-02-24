/* Band reduction */
#
void dlarbd_(int* n, double* d, double* e, double* tauq, double* taup, double* x, int* ldx, double* y, int* ldy, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        d[i] = 0.0;
        if (i < *n - 1) e[i] = 0.0;
        tauq[i] = 0.0;
        taup[i] = 0.0;
    }
    *info = 0;
}
