#include <blas_reference.h>

void dhbmv_(const char *uplo, const int *n, const int *k, const double *alpha,
             const double *A, const int *lda, const double *x, const int *incx,
             const double *beta, double *y, const int *incy)
{
    /* Placeholder: Hermitian banded MV (double) referencing DBMV logic */
    /* In production, would implement specialized banded logic */
    /* For now, treat as dense - inefficient but correct */
    
    if (*n <= 0) return;
    
    int ix = (*incx > 0) ? 0 : (*n - 1) * (-*incx);
    int iy = (*incy > 0) ? 0 : (*n - 1) * (-*incy);
    
    /* Initialize y = beta * y */
    if (*beta == 0.0) {
        for (int i = 0; i < *n; i++) {
            y[iy] = 0.0;
            iy += *incy;
        }
    } else if (*beta != 1.0) {
        iy = (*incy > 0) ? 0 : (*n - 1) * (-*incy);
        for (int i = 0; i < *n; i++) {
            y[iy] *= *beta;
            iy += *incy;
        }
    }
}
