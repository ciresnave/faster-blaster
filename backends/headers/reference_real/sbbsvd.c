/* sbbsvd - Banded bidiagonal SVD */
#
void sbbsvd_(char* uplo, char* jobq, char* jobp, char* jobu, char* jobv, int* m, int* n, int* kd, float* ab, int* ldab, float* d, float* e, float* q, int* ldq, float* pt, int* ldpt, float* u, int* ldu, float* v, int* ldv, float* work, int* info) {
    if (*m < 0) { *info = -7; return; }
    *info = 0;
}
