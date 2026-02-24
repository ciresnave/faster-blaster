/* Generalized RQ */

#
#include <stdlib.h>

void dggrqf_(const int *m, const int *p, const int *n, double *a, const int *lda,
             double *taua, double *b, const int *ldb, double *taub, double *work,
             const int *lwork, int *info)
{
    *info = 0;
    if (*m < 0) { *info = -1; return; }
    if (*p < 0) { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*lda < *m) { *info = -5; return; }
    if (*ldb < *p) { *info = -8; return; }
}
