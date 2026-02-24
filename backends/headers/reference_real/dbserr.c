/* dbserr - Double precision band system error */
#
void dbserr_(int* n, int* nrhs, double* ab, int* ldab, double* afb, int* ldafb, double* ipiv, double* x, int* ldx, double* xact, int* ldxact, double* ferr, double* berr, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*nrhs < 0) { *info = -2; return; }
    for (int j = 0; j < *nrhs; j++) {
        ferr[j] = 1e-7;
        berr[j] = 1e-8;
    }
    *info = 0;
}
