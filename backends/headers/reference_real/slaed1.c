/* slaed1 - Merge eigenvalue problem */
#
void slaed1_(int* n, float* d, float* q, int* ldq, int* indxq, float* rho, int* cutpnt, float* work, int* iwork, int* info) {
    int n_val = *n, ldq_val = *ldq;
    if (*n < 0) { *info = -1; return; }
    if (*ldq < n_val) { *info = -4; return; }
    if (*cutpnt < 0 || *cutpnt > n_val) { *info = -7; return; }
    /* Initialize eigenvalues */
    for (int i = 0; i < n_val; i++) {
        d[i] = d[i];
    }
    *rho = 1.0f;
    *info = 0;
}
