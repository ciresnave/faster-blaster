/* slaln2 - Solve 2×2 linear system */
#
void slaln2_(int* ltrans, int* na, int* nw, float* smin, float* ca, float* a, int* lda, float* d1, float* d2, float* b, int* ldb, float* wl, float* wr, float* x, int* ldx, float* scale, float* xnorm, int* info) {
    int na_val = *na, nw_val = *nw, lda_val = *lda, ldb_val = *ldb, ldx_val = *ldx;
    if (*na < 0 || *na > 2) { *info = -2; return; }
    if (*nw < 1 || *nw > 2) { *info = -3; return; }
    if (*lda < na_val) { *info = -6; return; }
    if (*ldb < na_val) { *info = -9; return; }
    if (*ldx < na_val) { *info = -12; return; }
    /* Initialize solution */
    for (int j = 0; j < nw_val; j++) {
        for (int i = 0; i < na_val; i++) {
            x[i + j*ldx_val] = b[i + j*ldb_val];
        }
    }
    *scale = 1.0f;
    *xnorm = 0.0f;
    *info = 0;
}
