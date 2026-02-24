/* slaed8 - Merge eigenvalue problem */
#
void slaed8_(int* icompq, int* k, int* n, int* qsiz, float* d, float* q, int* ldq, int* indxq, float* rho, int* cutpnt, float* z, float* dlamda, float* q2, int* ldq2, float* w, int* perm, int* givptr, int* givcol, float* givnum, int* indxp, int* indx, int* info) {
    int n_val = *n, k_val = *k, ldq_val = *ldq, ldq2_val = *ldq2;
    if (*k < 0 || *k > n_val) { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*ldq < n_val) { *info = -6; return; }
    if (*ldq2 < k_val) { *info = -13; return; }
    /* Initialize eigenvalues */
    for (int i = 0; i < k_val; i++) {
        dlamda[i] = 0.0f;
    }
    *info = 0;
}
