/* spotrc */
#
#include <float.h>
#include <string.h>
#include <blas_reference.h>

/* Cholesky solve with rcond estimation */
void spotrc_(const char *uplo, const int *n, float *A, const int *lda,
             float *work, float *rcond, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;

    if (n_val <= 0) return;

    /* Simplified: estimate condition number from diagonal */
    float dmin = FLT_MAX;
    float dmax = 0.0f;
    
    for (int i = 0; i < n_val; i++) {
        float d = A[i + i * lda_val];
        dmin = fminf(dmin, fabsf(d));
        dmax = fmaxf(dmax, fabsf(d));
    }

    if (dmin <= 0.0f) {
        *rcond = 0.0f;
    } else {
        *rcond = dmin / dmax;
    }
}
