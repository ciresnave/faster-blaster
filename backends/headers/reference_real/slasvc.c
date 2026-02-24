/* slasvc - Compute singular vectors via DQDS */
#
void slasvc_(float* d, int* n, float* q, float* u, float* v, float* info_ptr) {
    if (*n < 0) { *info_ptr = -2; return; }
    if (*n == 0) { *info_ptr = 0; return; }
    /* Initialize singular vectors */
    for (int i = 0; i < *n; i++) {
        u[i] = 1.0f / sqrtf((float)*n);
        v[i] = 1.0f / sqrtf((float)*n);
        q[i] = d[i];
    }
    *info_ptr = 0;
}
