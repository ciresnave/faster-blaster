/* Equilibrate symmetric */
#
#include <float.h>
#include <string.h>
#include <blas_reference.h>

/* Symmetric matrix equilibration - compute row scaling factors */
void syequ_(const char *uplo, const int *n, const float *A, const int *lda,
             float *s, float *scond, float *amax, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;
    *amax = 0.0f;

    if (n_val <= 0) {
        *scond = 1.0f;
        return;
    }

    int is_upper = (*uplo == 'U' || *uplo == 'u');
    float smin = FLT_MAX;
    float smax = 0.0f;

    /* For symmetric matrix, find max in each row/column */
    for (int i = 0; i < n_val; i++) {
        s[i] = 0.0f;
        
        if (is_upper) {
            /* Upper triangle */
            for (int j = 0; j <= i; j++) {
                s[i] = fmaxf(s[i], fabsf(A[j + i * lda_val]));
            }
        } else {
            /* Lower triangle */
            for (int j = i; j < n_val; j++) {
                s[i] = fmaxf(s[i], fabsf(A[j + i * lda_val]));
            }
        }
        
        *amax = fmaxf(*amax, s[i]);
        
        if (s[i] > 0.0f) {
            smin = fminf(smin, s[i]);
            smax = fmaxf(smax, s[i]);
        }
    }

    /* Compute scaling factors */
    if (*amax == 0.0f) {
        *scond = 1.0f;
    } else {
        for (int i = 0; i < n_val; i++) {
            if (s[i] > 0.0f) {
                s[i] = 1.0f / sqrtf(s[i]);
            } else {
                s[i] = 1.0f;
            }
        }
        
        *scond = (smin == 0.0f || smax == 0.0f) ? 1.0f : smin / smax;
    }
}
