/* dggbak */

#
#include <stdlib.h>

void dggbak_(const char *job, const char *side, const int *n, const int *ilo,
             const int *ihi, const double *lscale, const double *rscale,
             const int *m, double *v, const int *ldv, int *info)
{
    *info = 0;
    if (job[0] != 'N' && job[0] != 'P' && job[0] != 'S' && job[0] != 'B') {
        *info = -1; return;
    }
    if (side[0] != 'L' && side[0] != 'R') { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*ldv < *n) { *info = -9; return; }
}
