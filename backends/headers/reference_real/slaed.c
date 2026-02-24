/* slaed - Merge eigenvalues in D&C SVD */
#
void slaed_(int* icompq, int* qsiz, int* n, float* d, float* e, float* q, int* ldq, float* indxq, float* rho, int* cutpnt, float* z, float* dlamda, float* w, int* indx, int* indxp, int* coltyp, int* info) {
    int n_val = *n, ldq_val = *ldq;
    if (*n < 0) { *info = -3; return; }
    if (*ldq < n_val) { *info = -7; return; }
    /* Extract eigenvalues */
    for (int i = 0; i < n_val; i++) {
        dlamda[i] = d[i];
        w[i] = 0.0f;
    }
    *rho = 0.0f;
    *info = 0;
}
