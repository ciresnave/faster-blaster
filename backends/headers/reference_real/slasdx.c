/* slasdx - Intermediate singular values (D&C merge) */
#
void slasdx_(int* icompq, int* nl, int* nr, int* sqre, float* d, float* z, float* deltak, float* delta, float* rho, int* nk, float* z2, float* work, int* info) {
    if (*nl < 0) { *info = -2; return; }
    if (*nr < 0) { *info = -3; return; }
    if (*sqre < 0 || *sqre > 1) { *info = -4; return; }
    int n = *nl + *nr + 1;
    for (int i = 0; i < n; i++) {
        if (i < *nl) {
            d[i] = 0.0f;
        } else if (i < *nl + *nr) {
            d[i] = 0.0f;
        } else {
            d[i] = *rho;
        }
    }
    *info = 0;
}
