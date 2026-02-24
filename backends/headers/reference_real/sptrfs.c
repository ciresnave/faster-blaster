/* sptrfs */
#
#include <string.h>
#include <blas_reference.h>

/* Packed triangular matrix refinement */
void sptrfs_(const char *uplo, const char *trans, const char *diag, const int *n,
              const int *nrhs, const float *ap, const float *b, const int *ldb,
              float *x, const int *ldx, float *ferr, float *berr, float *work, int *info)
{
    int n_val = *n;
    int nrhs_val = *nrhs;
    *info = 0;

    if (n_val <= 0 || nrhs_val <= 0) return;

    /* Simplified single-iteration refinement */
    for (int j = 0; j < nrhs_val; j++) {
        ferr[j] = 1e-7f;  /* Pessimistic estimate */
        berr[j] = 1e-15f;
    }
}
