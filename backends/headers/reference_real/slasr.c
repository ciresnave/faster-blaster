/* Apply Givens */

#

/* SLASR - Apply sequence of plane rotations to matrix
   SIDE: 'L' apply from left, 'R' apply from right
   PIVOT: 'V' variable, 'T' top, 'B' bottom
   DIRECT: 'F' forward, 'B' backward
   M: Number of rows
   N: Number of columns
   C: Cosines
   S: Sines
   A: Matrix
   LDA: Leading dimension
*/
void slasr_(const char *side, const char *pivot, const char *direct, const int *m,
            const int *n, const float *c, const float *s, float *a, const int *lda)
{
    int m_val = *m;
    int n_val = *n;
    int lda_val = *lda;

    if (side[0] == 'L') {
        /* Apply rotations from left (to rows) */
        int istart, iend, iinc;
        if (direct[0] == 'F') {
            istart = 0;
            iend = m_val - 1;
            iinc = 1;
        } else {
            istart = m_val - 2;
            iend = 0;
            iinc = -1;
        }

        for (int i = istart; (iinc > 0) ? (i < iend) : (i > iend); i += iinc) {
            int i1, i2;
            if (pivot[0] == 'V') {
                i1 = i;
                i2 = i + 1;
            } else if (pivot[0] == 'T') {
                i1 = 0;
                i2 = i + 1;
            } else {
                i1 = i;
                i2 = m_val - 1;
            }

            float ci = c[i];
            float si = s[i];

            for (int j = 0; j < n_val; j++) {
                float tmp1 = ci * a[i1 + j * lda_val] + si * a[i2 + j * lda_val];
                float tmp2 = -si * a[i1 + j * lda_val] + ci * a[i2 + j * lda_val];
                a[i1 + j * lda_val] = tmp1;
                a[i2 + j * lda_val] = tmp2;
            }
        }
    } else {
        /* Apply rotations from right (to columns) */
        int jstart, jend, jinc;
        if (direct[0] == 'F') {
            jstart = 0;
            jend = n_val - 1;
            jinc = 1;
        } else {
            jstart = n_val - 2;
            jend = 0;
            jinc = -1;
        }

        for (int j = jstart; (jinc > 0) ? (j < jend) : (j > jend); j += jinc) {
            int j1, j2;
            if (pivot[0] == 'V') {
                j1 = j;
                j2 = j + 1;
            } else if (pivot[0] == 'T') {
                j1 = 0;
                j2 = j + 1;
            } else {
                j1 = j;
                j2 = n_val - 1;
            }

            float cj = c[j];
            float sj = s[j];

            for (int i = 0; i < m_val; i++) {
                float tmp1 = cj * a[i + j1 * lda_val] + sj * a[i + j2 * lda_val];
                float tmp2 = -sj * a[i + j1 * lda_val] + cj * a[i + j2 * lda_val];
                a[i + j1 * lda_val] = tmp1;
                a[i + j2 * lda_val] = tmp2;
            }
        }
    }
}
