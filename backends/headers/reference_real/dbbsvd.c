/* dbbsvd - Double precision banded bidiagonal SVD */
#
void dbbsvd_(char* uplo, char* jobq, char* jobp, char* jobu, char* jobv, int* m, int* n, int* kd, double* ab, int* ldab, double* d, double* e, double* q, int* ldq, double* pt, int* ldpt, double* u, int* ldu, double* v, int* ldv, double* work, int* info) {
    if (*m < 0) { *info = -7; return; }
    *info = 0;
}
