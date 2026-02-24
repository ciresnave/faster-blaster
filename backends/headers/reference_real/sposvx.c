#
#include <blas_reference.h>

void sposvx_(const int *m, const int *n, float *A, const int *lda,
             float *work, const int *lwork, int *info)
{
    /* Placeholder: Symmetric solve with equilibration */
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
}
