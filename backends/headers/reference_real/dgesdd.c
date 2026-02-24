#
#include <blas_reference.h>

void dgesdd_(const int *m, const int *n, double *A, const int *lda,
             double *work, const int *lwork, int *info)
{
    /* Placeholder: SVD (divide-and-conquer driver) */
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
}
