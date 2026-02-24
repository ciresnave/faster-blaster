/* dlasdx - Double precision intermediate singular values */
#
void dlasdx_(int* icompq, int* nl, int* nr, int* sqre, double* d, double* z, double* deltak, double* delta, double* rho, int* nk, double* z2, double* work, int* info) {
    if (*nl < 0) { *info = -2; return; }
    if (*nr < 0) { *info = -3; return; }
    if (*sqre < 0 || *sqre > 1) { *info = -4; return; }
    int n = *nl + *nr + 1;
    for (int i = 0; i < n; i++) {
        if (i < *nl) {
            d[i] = 0.0;
        } else if (i < *nl + *nr) {
            d[i] = 0.0;
        } else {
            d[i] = *rho;
        }
    }
    *info = 0;
}
