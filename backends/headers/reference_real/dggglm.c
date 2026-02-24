/* dggglm - Double precision generalized linear GLM solve */
#
void dggglm_(int* n, int* m, int* p, double* a, int* lda, double* b, int* ldb, double* d, double* x, double* y, double* work, int* lwork, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
