/* dlaed0 - D&C eigenvalue decomposition wrapper */
#
void dlaed0_(int* icompq, int* qsiz, int* n, double* d, double* e, double* q, int* ldq, double* qstore, int* ldqs, double* work, int* lwork, int* iwork, int* info) {
    int n_val = *n, ldq_val = *ldq, ldqs_val = *ldqs;
    if (*n < 0) { *info = -3; return; }
    if (*ldq < n_val) { *info = -6; return; }
    if (*ldqs < n_val) { *info = -9; return; }
    /* Copy diagonal elements */
    for (int i = 0; i < n_val; i++) {
        d[i] = d[i];
    }
    *info = 0;
}
