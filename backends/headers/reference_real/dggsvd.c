/* Generalized SVD */

#
#include <stdlib.h>

void dggsvd_(const char *jobu, const char *jobv, const char *jobq, const int *m,
             const int *n, const int *p, int *k, int *l, double *a, const int *lda,
             double *b, const int *ldb, double *alpha, double *beta, double *u,
             const int *ldu, double *v, const int *ldv, double *q, const int *ldq,
             double *work, int *iwork, int *info)
{
    *info = 0;
    *k = 0;
    *l = 0;
}
