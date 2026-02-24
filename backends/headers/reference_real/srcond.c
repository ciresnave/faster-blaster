/* srcond - Auxiliary reciprocal condition number */
#
void srcond_(int* n, float* a, int* lda, float* anorm, float* rcond, int* info) {
    if (*n <= 0) { *info = -1; return; }
    *info = 0;
}
