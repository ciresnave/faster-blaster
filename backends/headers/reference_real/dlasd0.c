/* Divide-and-conquer SVD */

#
#include <stdlib.h>

void dlasd0_(const int *n, const int *sqre, double *d, const double *e,
             double *u, const int *ldu, double *vt, const int *ldvt,
             const int *smlsiz, int *iwork, double *work, int *info)
{
    *info = 0;
    if (*n < 0) { *info = -1; return; }
}
