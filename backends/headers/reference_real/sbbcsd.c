/* sbbcsd - Banded Bidiagonal CS decomposition */
#
void sbbcsd_(char* jobu1, char* jobu2, char* jobv1t, char* jobv2t, char* trans, int* m, int* q, int* p, float* theta, float* phi, float* u1, int* ldu1, float* u2, int* ldu2, float* v1t, int* ldv1t, float* v2t, int* ldv2t, float* work, int* lwork, int* info) {
    if (*m < 0) { *info = -7; return; }
    *info = 0;
}
