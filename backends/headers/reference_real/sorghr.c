#
#include <blas_reference.h>

void sorghr_(const int *m, const int *n, float *A, const int *lda,
             float *work, const int *lwork, int *info)
{
    /* Placeholder: Generate Householder Q from Hessenberg reduction */
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
}
