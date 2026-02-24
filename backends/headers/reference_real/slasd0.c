/* Divide-and-conquer SVD */

#
#include <stdlib.h>

void slasd0_(const int *n, const int *sqre, float *d, const float *e,
             float *u, const int *ldu, float *vt, const int *ldvt,
             const int *smlsiz, int *iwork, float *work, int *info)
{
    *info = 0;
    if (*n < 0) { *info = -1; return; }
}
