/* dlahz - Double precision Hessenberg similarity */
#
void dlahz_(int* n, double* a, int* lda, double* h, int* ldh, double* tau, double* c, int* ldc, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*lda < *n) { *info = -3; return; }
    if (*ldh < *n) { *info = -5; return; }
    if (*ldc < *n) { *info = -8; return; }
    *info = 0;
}
