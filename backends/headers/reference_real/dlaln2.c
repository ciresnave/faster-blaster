/* dlaln2 - Solve 2×2 linear system */
#
void dlaln2_(int* ltrans, int* na, int* nw, double* smin, double* ca, double* a, int* lda, double* d1, double* d2, double* b, int* ldb, double* wl, double* wr, double* x, int* ldx, double* scale, double* xnorm, int* info) {
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
    *scale = 1.0;
    *xnorm = 0.0;
    *info = 0;
}
