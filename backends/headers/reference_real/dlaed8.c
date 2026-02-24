/* dlaed8 - Merge eigenvalue problem */
#
void dlaed8_(int* icompq, int* k, int* n, int* qsiz, double* d, double* q, int* ldq, int* indxq, double* rho, int* cutpnt, double* z, double* dlamda, double* q2, int* ldq2, double* w, int* perm, int* givptr, int* givcol, double* givnum, int* indxp, int* indx, int* info) {
    int n_val = *n, k_val = *k, ldq_val = *ldq, ldq2_val = *ldq2;
    if (*k < 0 || *k > n_val) { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*ldq < n_val) { *info = -6; return; }
    if (*ldq2 < k_val) { *info = -13; return; }
    /* Initialize eigenvalues */
    for (int i = 0; i < k_val; i++) {
        dlamda[i] = 0.0;
    }
    *info = 0;
}
