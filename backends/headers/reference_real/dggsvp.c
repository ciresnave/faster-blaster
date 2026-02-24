/* Generalized SVD prep */

#
#include <stdlib.h>

void dggsvp_(const char *jobu, const char *jobv, const char *jobq, const int *m,
             const int *p, const int *n, double *a, const int *lda, double *b,
             const int *ldb, const double *tola, const double *tolb,
             int *k, int *l, double *u, const int *ldu, double *v, const int *ldv,
             double *q, const int *ldq, int *iwork, double *tau, double *work, int *info)
{
    *info = 0;
    *k = 0;
    *l = 0;
}
