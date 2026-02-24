/* Generalized RQ */

#
#include <stdlib.h>

void sggrqf_(const int *m, const int *p, const int *n, float *a, const int *lda,
             float *taua, float *b, const int *ldb, float *taub, float *work,
             const int *lwork, int *info)
{
    *info = 0;
    if (*m < 0) { *info = -1; return; }
    if (*p < 0) { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*lda < *m) { *info = -5; return; }
    if (*ldb < *p) { *info = -8; return; }
}
