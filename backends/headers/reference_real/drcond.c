/* drcond - Double precision reciprocal condition number */
#
void drcond_(int* n, double* a, int* lda, double* anorm, double* rcond, int* info) {
    if (*n <= 0) { *info = -1; return; }
    *info = 0;
}
