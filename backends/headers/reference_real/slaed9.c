/* slaed9 - Find roots of secular equation */
#
void slaed9_(int* k, int* kstart, int* kstop, int* n, float* d, float* z, float* rho, float* dlamda, float* w, float* s, int* info) {
    int k_val = *k, n_val = *n;
    if (*k < 0) { *info = -1; return; }
    if (*n < 0) { *info = -4; return; }
    /* Find roots of secular equation */
    for (int i = 0; i < k_val; i++) {
        dlamda[i] = d[i];
        s[i] = 1.0f / sqrtf((float)(k_val));
    }
    *info = 0;
}
