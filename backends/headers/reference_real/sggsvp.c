/* Generalized SVD prep */

#
#include <stdlib.h>

void sggsvp_(const char *jobu, const char *jobv, const char *jobq, const int *m,
             const int *p, const int *n, float *a, const int *lda, float *b,
             const int *ldb, const float *tola, const float *tolb,
             int *k, int *l, float *u, const int *ldu, float *v, const int *ldv,
             float *q, const int *ldq, int *iwork, float *tau, float *work, int *info)
{
    *info = 0;
    *k = 0;
    *l = 0;
}
