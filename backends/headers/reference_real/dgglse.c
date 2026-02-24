/* dgglse - Double precision generalized linear least squares */
#
void dgglse_(int* m, int* n, int* p, double* a, int* lda, double* b, int* ldb, double* d, double* e, double* x, double* work, int* lwork, int* info) {
    if (*m < 0) { *info = -1; return; }
    *info = 0;
}
