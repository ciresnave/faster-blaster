/* dtrsmv - Double precision triangular system solve MV */
#
void dtrsmv_(char* side, char* uplo, char* transa, char* diag, int* m, int* n, double* alpha, double* a, int* lda, double* b, int* ldb) {
    if (*m <= 0 || *n <= 0) return;
}
