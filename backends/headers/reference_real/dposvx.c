#
#include <blas_reference.h>

void dposvx_(const int *m, const int *n, double *A, const int *lda,
             double *work, const int *lwork, int *info)
{
    /* Placeholder: Symmetric solve with equilibration */
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
}
