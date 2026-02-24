/* slalsa - Left singular vector update via bisection */
#
void slalsa_(int* icompq, int* smlsiz, int* n, int* nrhs, float* b, int* ldb, float* bx, int* ldbx, float* u, int* ldu, float* vt, int* ldvt, int* k, float* difl, float* difr, float* z, float* poles, int* givcol, int* ldgcol, int* perm, float* givnum, float* c, float* s, float* work, int* lwork, int* iwork, int* info) {
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
