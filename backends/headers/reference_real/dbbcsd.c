/* dbbcsd - Double precision banded bidiagonal CS decomposition */
#
void dbbcsd_(char* jobu1, char* jobu2, char* jobv1t, char* jobv2t, char* trans, int* m, int* q, int* p, double* theta, double* phi, double* u1, int* ldu1, double* u2, int* ldu2, double* v1t, int* ldv1t, double* v2t, int* ldv2t, double* work, int* lwork, int* info) {
    if (*m < 0) { *info = -7; return; }
    *info = 0;
}
