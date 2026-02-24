/* Apply Householder */

#

/* SLARF - Apply Householder reflection to rows/columns of matrix
   SIDE: 'L' apply from left, 'R' apply from right
   M: Number of rows
   N: Number of columns
   V: Householder vector
   INCV: Increment between V elements
   TAU: Scalar factor
   C: Matrix to apply to
   LDC: Leading dimension of C
   WORK: Workspace
*/
void slarf_(const char *side, const int *m, const int *n, const float *v,
            const int *incv, const float *tau, float *c, const int *ldc, float *work)
{
    int m_val = *m;
    int n_val = *n;
    int ldc_val = *ldc;
    int incv_val = *incv;
    float tau_val = *tau;

    if (tau_val == 0.0f) return;

    if (side[0] == 'L') {
        /* Apply H from the left: C := H * C */
        /* w := C^T * v */
        for (int j = 0; j < n_val; j++) {
            work[j] = 0.0f;
        }

        /* Compute w = C^T * v */
        for (int i = 0; i < m_val; i++) {
            float vi = v[i * incv_val];
            for (int j = 0; j < n_val; j++) {
                work[j] += vi * c[i + j * ldc_val];
            }
        }

        /* C := C - tau * v * w^T */
        for (int i = 0; i < m_val; i++) {
            float vi = tau_val * v[i * incv_val];
            for (int j = 0; j < n_val; j++) {
                c[i + j * ldc_val] -= vi * work[j];
            }
        }
    } else {
        /* Apply H from the right: C := C * H */
        /* w := C * v */
        for (int i = 0; i < m_val; i++) {
            work[i] = 0.0f;
        }

        /* Compute w = C * v */
        for (int j = 0; j < n_val; j++) {
            float vj = v[j * incv_val];
            for (int i = 0; i < m_val; i++) {
                work[i] += c[i + j * ldc_val] * vj;
            }
        }

        /* C := C - w * tau * v^T */
        for (int j = 0; j < n_val; j++) {
            float vj = tau_val * v[j * incv_val];
            for (int i = 0; i < m_val; i++) {
                c[i + j * ldc_val] -= work[i] * vj;
            }
        }
    }
}
