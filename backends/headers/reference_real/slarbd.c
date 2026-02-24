/* Band reduction */
#
void slarbd_(int* n, float* d, float* e, float* tauq, float* taup, float* x, int* ldx, float* y, int* ldy, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        d[i] = 0.0f;
        if (i < *n - 1) e[i] = 0.0f;
        tauq[i] = 0.0f;
        taup[i] = 0.0f;
    }
    *info = 0;
}
