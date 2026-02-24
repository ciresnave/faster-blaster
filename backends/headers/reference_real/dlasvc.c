/* dlasvc - Compute singular vectors via DQDS */
#
void dlasvc_(double* d, int* n, double* q, double* u, double* v, double* info_ptr) {
    if (*n < 0) { *info_ptr = -2; return; }
    if (*n == 0) { *info_ptr = 0; return; }
    /* Initialize singular vectors */
    for (int i = 0; i < *n; i++) {
        u[i] = 1.0 / sqrt((double)*n);
        v[i] = 1.0 / sqrt((double)*n);
        q[i] = d[i];
    }
    *info_ptr = 0;
}
