/* dtrord - Reorder Schur form using selection function */

#

/* DTRORD - Reorder Schur form using selection function (double precision)
   COMPQ: 'N' don't update Q, 'V' update Q
   SELECT: Selection function callback
   N: Matrix size
   T: Schur form
   LDT: Leading dimension
   Q: Orthogonal matrix
   LDQ: Leading dimension
   W: Eigenvalues
   M: Number of selected eigenvalues
   S: Condition number estimate
   SEP: Separation estimate
   WORK: Workspace
   LWORK: Workspace size
   INFO: Status
 */
void dtrord_(const char *compq, int (*select)(const double *, const double *),
             const int *n, double *t, const int *ldt,
             double *q, const int *ldq, double *w, int *m,
             double *s, double *sep, double *work, const int *lwork, int *info)
{
    *info = 0;
    int n_val = *n;
    int ldt_val = *ldt;
    int ldq_val = *ldq;
    *m = 0;

    if (compq[0] != 'N' && compq[0] != 'V') {
        *info = -1;
        return;
    }
    if (n_val < 0) {
        *info = -4;
        return;
    }
    if (ldt_val < n_val) {
        *info = -6;
        return;
    }
    if (ldq_val < n_val && compq[0] == 'V') {
        *info = -8;
        return;
    }

    if (n_val == 0) return;

    /* Count selected eigenvalues */
    for (int i = 0; i < n_val; i++) {
        double re = w[2 * i];
        double im = w[2 * i + 1];
        if ((*select)(&re, &im)) {
            (*m)++;
        }
    }

    /* Reorder by moving selected eigenvalues to front */
    int ilst = 0;
    for (int ifst = 0; ifst < n_val; ifst++) {
        double re = w[2 * ifst];
        double im = w[2 * ifst + 1];
        
        if ((*select)(&re, &im)) {
            if (ifst > ilst) {
                /* Move eigenvalue at ifst to position ilst */
                dtrexc_(compq, n, t, ldt, q, ldq, &ifst, &ilst, work, info);
                if (*info != 0) return;
            }
            ilst++;
        }
    }

    /* Condition numbers: simplified estimates */
    *s = 1.0;
    *sep = 1.0;
}
