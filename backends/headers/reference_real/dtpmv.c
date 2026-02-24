#include <blas_reference.h>

void dtpmv_(const char *uplo, const int *n, const double *alpha,
             const double *AP, const double *x, const int *incx,
             const double *beta, double *y, const int *incy)
{
    /* Placeholder: Packed triangular MV (packed storage) */
    /* Reference: src/blas/level2/dtrmv.c */
    if (*n <= 0) return;
}
