/* dggbal */

#
#include <stdlib.h>

void dggbal_(const char *job, const int *n, double *a, const int *lda,
             double *b, const int *ldb, int *ilo, int *ihi, double *lscale,
             double *rscale, double *work, int *info)
{
    *info = 0;
    if (job[0] != 'N' && job[0] != 'P' && job[0] != 'S' && job[0] != 'B') {
        *info = -1; return;
    }
    if (*n < 0) { *info = -2; return; }
    if (*lda < *n) { *info = -4; return; }
    if (*ldb < *n) { *info = -6; return; }
    
    *ilo = 1;
    *ihi = *n;
    for (int i = 0; i < *n; i++) {
        lscale[i] = 1.0;
        rscale[i] = 1.0;
    }
}
