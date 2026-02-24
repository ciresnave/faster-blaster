/* slaed2 - Merge eigenvalues for subproblems */
#
void slaed2_(int* k, int* n, int* n1, float* d, float* q, int* ldq, int* indxq, float* rho, float* z, float* dlamda, float* w, int* indx, int* indxc, int* indxp, int* coltyp, int* info) {
    int n_val = *n, ldq_val = *ldq;
    if (*n < 0) { *info = -2; return; }
    if (*ldq < n_val) { *info = -5; return; }
    /* Initialize k eigenvalue count */
    *k = 0;
    for (int i = 0; i < n_val; i++) {
        if (d[i] > 0.0f) (*k)++;
    }
    *info = 0;
}
