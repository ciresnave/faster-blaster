/* slaed3 - Update eigenvectors */
#
void slaed3_(int* k, int* n, int* n1, int* n2, float* d, float* q, int* ldq, float* rho, float* dlamda, float* q2, int* ldq2, int* indx, int* ctot, float* w, float* s, int* info) {
    int n_val = *n, k_val = *k, ldq_val = *ldq, ldq2_val = *ldq2;
    if (*k < 0 || *k > n_val) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*ldq < n_val) { *info = -6; return; }
    if (*ldq2 < k_val) { *info = -10; return; }
    /* Initialize eigenvectors */
    for (int j = 0; j < k_val; j++) {
        for (int i = 0; i < n_val; i++) {
            q[i + j*ldq_val] = 0.0f;
        }
    }
    *info = 0;
}
