/* strexc - Reorder Schur form by swapping diagonal blocks */

#

/* STREXC - Reorder Schur form T by swapping adjacent blocks
   COMPQ: 'N' don't update Q, 'V' update Q
   N: Matrix size
   T: Schur form matrix (upper triangular)
   LDT: Leading dimension of T
   Q: Orthogonal matrix (if COMPQ='V')
   LDQ: Leading dimension of Q
   IFST: First row of block to move (1-based)
   ILST: Destination row (1-based)
   WORK: Workspace
   INFO: Status
 */
void strexc_(const char *compq, const int *n, float *t, const int *ldt,
             float *q, const int *ldq, int *ifst, int *ilst, float *work,
             int *info)
{
    *info = 0;
    int n_val = *n;
    int ldt_val = *ldt;
    int ldq_val = *ldq;
    int ifst_val = *ifst - 1;  /* Convert to 0-based */
    int ilst_val = *ilst - 1;

    if (compq[0] != 'N' && compq[0] != 'V') {
        *info = -1;
        return;
    }
    if (n_val < 0) {
        *info = -2;
        return;
    }
    if (ldt_val < n_val) {
        *info = -4;
        return;
    }
    if (ldq_val < n_val && compq[0] == 'V') {
        *info = -6;
        return;
    }
    if (ifst_val < 0 || ifst_val >= n_val || ilst_val < 0 || ilst_val >= n_val) {
        *info = -7;
        return;
    }

    if (ifst_val == ilst_val) return;

    /* Direction: forward if moving down, backward if moving up */
    int dir = (ilst_val > ifst_val) ? 1 : -1;
    int idx_start = (dir > 0) ? ifst_val : ilst_val;
    int idx_end = (dir > 0) ? ilst_val : ifst_val;

    for (int idx = idx_start; (dir > 0) ? (idx < idx_end) : (idx < idx_end); idx += dir) {
        int i = (dir > 0) ? idx : idx + 1;
        int j = (dir > 0) ? idx + 1 : idx;

        /* Swap rows i and j, then columns i and j */
        /* Use Givens rotation to achieve swap */
        float c = 1.0f, s = 0.0f;

        /* Find Givens rotation to eliminate T[j,i] */
        if (i < j) {
            float a = t[i + i * ldt_val];
            float b = t[j + i * ldt_val];
            float r = sqrtf(a * a + b * b);
            if (r > 0.0f) {
                c = a / r;
                s = -b / r;
            }
        }

        /* Apply rotation to T and Q */
        for (int col = i; col < n_val; col++) {
            float tmp = c * t[i + col * ldt_val] - s * t[j + col * ldt_val];
            t[j + col * ldt_val] = s * t[i + col * ldt_val] + c * t[j + col * ldt_val];
            t[i + col * ldt_val] = tmp;
        }

        for (int row = 0; row <= j; row++) {
            float tmp = c * t[row + i * ldt_val] + s * t[row + j * ldt_val];
            t[row + j * ldt_val] = -s * t[row + i * ldt_val] + c * t[row + j * ldt_val];
            t[row + i * ldt_val] = tmp;
        }

        /* Update Q if requested */
        if (compq[0] == 'V') {
            for (int row = 0; row < n_val; row++) {
                float tmp = c * q[row + i * ldq_val] - s * q[row + j * ldq_val];
                q[row + j * ldq_val] = s * q[row + i * ldq_val] + c * q[row + j * ldq_val];
                q[row + i * ldq_val] = tmp;
            }
        }
    }

    *ifst = ifst_val + 1;  /* Convert back to 1-based */
    *ilst = ilst_val + 1;
}
