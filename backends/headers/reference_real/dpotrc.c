/* dpotrc */
#
#include <float.h>
#include <string.h>
#include <blas_reference.h>

/* Cholesky solve with rcond estimation */
void dpotrc_(const char *uplo, const int *n, double *A, const int *lda,
             double *work, double *rcond, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;

    if (n_val <= 0) return;

    /* Simplified: estimate condition number from diagonal */
    double dmin = DBL_MAX;
    double dmax = 0.0;
    
    for (int i = 0; i < n_val; i++) {
        double d = A[i + i * lda_val];
        dmin = fmin(dmin, fabs(d));
        dmax = fmax(dmax, fabs(d));
    }

    if (dmin <= 0.0) {
        *rcond = 0.0;
    } else {
        *rcond = dmin / dmax;
    }
}
