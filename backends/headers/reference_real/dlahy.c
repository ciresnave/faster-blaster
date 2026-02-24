/* dlahy - Double precision Householder on Hermitian */
#
void dlahy_(int* n, double* tau, double* c, double* x, int* ldx, double* h, int* ldh, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*ldx < *n) { *info = -5; return; }
    if (*ldh < *n) { *info = -7; return; }
    *info = 0;
}
