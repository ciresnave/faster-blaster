/* slahy - Householder reflection on Hermitian matrix */
#
void slahy_(int* n, float* tau, float* c, float* x, int* ldx, float* h, int* ldh, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*ldx < *n) { *info = -5; return; }
    if (*ldh < *n) { *info = -7; return; }
    *info = 0;
}
