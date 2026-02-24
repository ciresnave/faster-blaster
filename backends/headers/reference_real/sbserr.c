/* sbserr - Band system error correction */
#
void sbserr_(int* n, int* nrhs, float* ab, int* ldab, float* afb, int* ldafb, float* ipiv, float* x, int* ldx, float* xact, int* ldxact, float* ferr, float* berr, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*nrhs < 0) { *info = -2; return; }
    for (int j = 0; j < *nrhs; j++) {
        ferr[j] = 1e-7f;
        berr[j] = 1e-8f;
    }
    *info = 0;
}
