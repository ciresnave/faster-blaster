/* dlalsa - Left singular vector update via bisection */
#
void dlalsa_(int* icompq, int* smlsiz, int* n, int* nrhs, double* b, int* ldb, double* bx, int* ldbx, double* u, int* ldu, double* vt, int* ldvt, int* k, double* difl, double* difr, double* z, double* poles, int* givcol, int* ldgcol, int* perm, double* givnum, double* c, double* s, double* work, int* lwork, int* iwork, int* info) {
    if (*n < 0) { *info = -3; return; }
    if (*nrhs < 0) { *info = -4; return; }
    if (*ldb < *n) { *info = -6; return; }
    /* Initialize b -> bx */
    for (int j = 0; j < *nrhs; j++) {
        for (int i = 0; i < *n; i++) {
            bx[i + j*(*ldbx)] = b[i + j*(*ldb)];
        }
    }
    *info = 0;
}
