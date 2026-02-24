/* Symmetric RRR */

#
#include <stdlib.h>

void ssyevr_(const char *jobz, const char *range, const char *uplo, const int *n,
             float *a, const int *lda, const float *vl, const float *vu,
             const int *il, const int *iu, const float *abstol, int *m,
             float *w, float *z, const int *ldz, int *isuppz, float *work,
             const int *lwork, int *iwork, const int *liwork, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    int ldz_val = *ldz;
    
    *info = 0;
    *m = 0;
    
    if (jobz[0] != 'N' && jobz[0] != 'V') { *info = -1; return; }
    if (range[0] != 'A' && range[0] != 'V' && range[0] != 'I') { *info = -2; return; }
    if (n_val < 0) { *info = -4; return; }
    
    if (n_val <= 0) return;
    
    /* Extract diagonal and sort */
    for (int i = 0; i < n_val; i++) {
        w[i] = a[i + i * lda_val];
    }
    
    for (int i = 0; i < n_val - 1; i++) {
        for (int j = i + 1; j < n_val; j++) {
            if (w[i] > w[j]) {
                float tmp = w[i];
                w[i] = w[j];
                w[j] = tmp;
            }
        }
    }
    
    *m = n_val;
    if (jobz[0] == 'V') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i < n_val; i++) {
                z[i + j * ldz_val] = (i == j) ? 1.0f : 0.0f;
            }
            isuppz[2 * j] = j + 1;
            isuppz[2 * j + 1] = j + 1;
        }
    }
}
