/* sgglse - Generalized linear least squares solve */
#
void sgglse_(int* m, int* n, int* p, float* a, int* lda, float* b, int* ldb, float* d, float* e, float* x, float* work, int* lwork, int* info) {
    if (*m < 0) { *info = -1; return; }
    *info = 0;
}
