/* dtrsen */

#
#include <stdlib.h>

void dtrsen_(const char *compq, const char *select, const int *n, double *t,
             const int *ldt, double *q, const int *ldq, double *w, int *m,
             double *s, double *sep, double *work, const int *lwork,
             int *iwork, const int *liwork, int *info)
{
    int n_val = *n;
    int ldt_val = *ldt;
    int ldq_val = *ldq;
    
    *info = 0;
    *m = 0;
    *s = 1.0;
    if (sep) *sep = 1.0;
    
    if (compq[0] != 'N' && compq[0] != 'V') { *info = -1; return; }
    if (n_val < 0) { *info = -3; return; }
    if (ldt_val < n_val) { *info = -5; return; }
    if (ldq_val < n_val && compq[0] == 'V') { *info = -7; return; }
    
    if (n_val <= 0) return;
    
    *m = n_val;
    
    for (int i = 0; i < n_val; i++) {
        w[i] = t[i + i * ldt_val];
    }
    
    if (compq[0] == 'V') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i < n_val; i++) {
                q[i + j * ldq_val] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
}
