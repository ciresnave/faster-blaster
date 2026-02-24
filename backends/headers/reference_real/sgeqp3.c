#
#include <string.h>
#include <blas_reference.h>

/* QR factorization with column pivoting */
void sgeqp3_(const int *m, const int *n, float *A, const int *lda,
              int *jpvt, float *tau, float *work, const int *lwork, int *info)
{
    int m_val = *m;
    int n_val = *n;
    int lda_val = *lda;
    int lwork_val = *lwork;
    *info = 0;

    if (m_val <= 0 || n_val <= 0) return;

    /* Initialize pivot array if needed */
    for (int j = 0; j < n_val; j++) {
        if (jpvt[j] < 0) jpvt[j] = j + 1;  /* Convert to 1-based */
    }

    /* Perform QR factorization with pivoting (simplified: basic QR) */
    int k = (m_val < n_val) ? m_val : n_val;
    
    for (int i = 0; i < k; i++) {
        /* Find column with maximum norm for pivoting */
        float max_norm = 0.0f;
        int max_col = i;
        for (int j = i; j < n_val; j++) {
            float col_norm = 0.0f;
            for (int row = i; row < m_val; row++) {
                float val = A[row + j * lda_val];
                col_norm += val * val;
            }
            if (col_norm > max_norm) {
                max_norm = col_norm;
                max_col = j;
            }
        }

        /* Swap columns if needed */
        if (max_col != i) {
            for (int row = 0; row < m_val; row++) {
                float tmp = A[row + i * lda_val];
                A[row + i * lda_val] = A[row + max_col * lda_val];
                A[row + max_col * lda_val] = tmp;
            }
            int tmp_piv = jpvt[i];
            jpvt[i] = jpvt[max_col];
            jpvt[max_col] = tmp_piv;
        }

        /* Compute Householder reflection for column i */
        float sigma = 0.0f;
        for (int row = i; row < m_val; row++) {
            sigma += A[row + i * lda_val] * A[row + i * lda_val];
        }
        sigma = sqrtf(sigma);
        if (A[i + i * lda_val] > 0) sigma = -sigma;

        float alpha = A[i + i * lda_val] - sigma;
        tau[i] = -sigma / alpha;
        for (int row = i; row < m_val; row++) {
            A[row + i * lda_val] /= alpha;
        }
        A[i + i * lda_val] = sigma;

        /* Apply Householder reflection to remaining columns */
        for (int j = i + 1; j < n_val; j++) {
            float dot = 0.0f;
            for (int row = i; row < m_val; row++) {
                dot += A[row + i * lda_val] * A[row + j * lda_val];
            }
            dot *= tau[i];
            for (int row = i; row < m_val; row++) {
                A[row + j * lda_val] -= dot * A[row + i * lda_val];
            }
        }
    }
}
