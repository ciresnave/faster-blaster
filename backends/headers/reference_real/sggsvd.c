/* Generalized SVD */

#
#include <stdlib.h>

void sggsvd_(const char *jobu, const char *jobv, const char *jobq, const int *m,
             const int *n, const int *p, int *k, int *l, float *a, const int *lda,
             float *b, const int *ldb, float *alpha, float *beta, float *u,
             const int *ldu, float *v, const int *ldv, float *q, const int *ldq,
             float *work, int *iwork, int *info)
{
    *info = 0;
    *k = 0;
    *l = 0;
}
