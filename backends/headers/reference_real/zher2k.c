#include <blas_reference.h>

void zher2k_(const char *side, const char *uplo, const int *m, const int *n,
             const int *alpha, const int *A, const int *lda,
             const int *B, const int *ldb, const int *beta,
             int *C, const int *ldc)
{
    /* Placeholder: Hermitian rank-2k update */
    if (*m <= 0 || *n <= 0) return;
}
