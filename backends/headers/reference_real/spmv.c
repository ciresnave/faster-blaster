#include <blas_reference.h>

void spmv_(const char *uplo, const int *n, const float *alpha,
             const float *AP, const float *x, const int *incx,
             const float *beta, float *y, const int *incy)
{
    /* Placeholder: Packed symmetric MV (packed storage) */
    /* Reference: src/blas/level2/ssymv.c */
    if (*n <= 0) return;
}
