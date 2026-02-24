/* slahz - Hessenberg similarity transformation */
#
void slahz_(int* n, float* a, int* lda, float* h, int* ldh, float* tau, float* c, int* ldc, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*lda < *n) { *info = -3; return; }
    if (*ldh < *n) { *info = -5; return; }
    if (*ldc < *n) { *info = -8; return; }
    *info = 0;
}
