/* sggglm - Generalized linear GLM solve */
#
void sggglm_(int* n, int* m, int* p, float* a, int* lda, float* b, int* ldb, float* d, float* x, float* y, float* work, int* lwork, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
