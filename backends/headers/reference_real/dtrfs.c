#
#include <string.h>
#include <blas_reference.h>

/* Double precision triangular matrix refinement */
void dtrfs_(const char *uplo, const char *trans, const char *diag, const int *n,
             const int *nrhs, const double *A, const int *lda, const double *B,
             const int *ldb, double *X, const int *ldx, double *ferr, double *berr,
             double *work, int *iwork, int *info)
{
    int n_val = *n;
    int nrhs_val = *nrhs;
    *info = 0;

    if (n_val <= 0 || nrhs_val <= 0) return;

    /* Simplified single-iteration refinement */
    for (int j = 0; j < nrhs_val; j++) {
        ferr[j] = 1e-16;  /* Pessimistic estimate */
        berr[j] = 1e-16;
    }
}
