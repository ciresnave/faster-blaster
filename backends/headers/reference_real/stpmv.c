#include <blas_reference.h>

void stpmv_(const char *uplo, const int *n, const float *alpha,
             const float *AP, const float *x, const int *incx,
             const float *beta, float *y, const int *incy)
{
    /* Placeholder: Packed triangular MV (packed storage) */
    /* Reference: src/blas/level2/strmv.c */
    if (*n <= 0) return;
}
