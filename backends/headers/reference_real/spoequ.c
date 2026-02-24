/* Equilibrate Cholesky */
#
#include <float.h>
#include <string.h>
#include <blas_reference.h>

/* Positive definite matrix equilibration */
void spoequ_(const int *n, const float *A, const int *lda, float *s,
              float *scond, float *amax, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;
    *amax = 0.0f;

    if (n_val <= 0) {
        *scond = 1.0f;
        return;
    }

    /* For diagonal matrix, scaling factors based on sqrt of diagonal */
    float smin = FLT_MAX;
    float smax = 0.0f;

    for (int i = 0; i < n_val; i++) {
        s[i] = A[i + i * lda_val];
        *amax = fmaxf(*amax, s[i]);
        
        if (s[i] <= 0.0f) {
            *info = i + 1;
            return;
        }
        
        if (s[i] > 0.0f) {
            smin = fminf(smin, s[i]);
            smax = fmaxf(smax, s[i]);
        }
    }

    /* Compute scaling factors */
    for (int i = 0; i < n_val; i++) {
        s[i] = 1.0f / sqrtf(s[i]);
    }
    
    *scond = sqrtf(smin / smax);
}
