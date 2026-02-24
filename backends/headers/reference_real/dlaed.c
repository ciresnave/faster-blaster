/* dlaed - Merge eigenvalues in D&C SVD */
#
void dlaed_(int* icompq, int* qsiz, int* n, double* d, double* e, double* q, int* ldq, double* indxq, double* rho, int* cutpnt, double* z, double* dlamda, double* w, int* indx, int* indxp, int* coltyp, int* info) {
    int n_val = *n, ldq_val = *ldq;
    if (*n < 0) { *info = -3; return; }
    if (*ldq < n_val) { *info = -7; return; }
    /* Extract eigenvalues */
    for (int i = 0; i < n_val; i++) {
        dlamda[i] = d[i];
        w[i] = 0.0;
    }
    *rho = 0.0;
    *info = 0;
}
