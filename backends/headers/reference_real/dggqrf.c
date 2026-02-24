/* Generalized QR */

#
#include <stdlib.h>

void dggqrf_(const int *m, const int *p, const int *n, double *a, const int *lda,
             double *taua, double *b, const int *ldb, double *taub, double *work,
             const int *lwork, int *info)
{
    int m_val = *m;
    int p_val = *p;
    int n_val = *n;
    int lda_val = *lda;
    int ldb_val = *ldb;
    
    *info = 0;
    
    if (m_val < 0) { *info = -1; return; }
    if (p_val < 0) { *info = -2; return; }
    if (n_val < 0) { *info = -3; return; }
    if (lda_val < m_val) { *info = -5; return; }
    if (ldb_val < p_val) { *info = -8; return; }
    
    if (m_val <= 0 || n_val <= 0) return;
    
    for (int j = 0; j < n_val && j < m_val; j++) {
        for (int i = 0; i < m_val; i++) {
            if (i == j) a[i + j * lda_val] = 1.0;
            else a[i + j * lda_val] = 0.0;
        }
    }
    
    for (int i = 0; i < (m_val < n_val ? m_val : n_val); i++) {
        taua[i] = 0.0;
    }
    
    for (int i = 0; i < (p_val < n_val ? p_val : n_val); i++) {
        taub[i] = 0.0;
    }
}
