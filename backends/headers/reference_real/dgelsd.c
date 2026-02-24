#
#include <blas_reference.h>

void dgelsd_(const int *m, const int *n, double *A, const int *lda,
             double *work, const int *lwork, int *info)
{
    /* Placeholder: Least squares SVD (driver) */
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
}
