/* dlaed6 - Solve secular equation for general n */
#
void dlaed6_(int* kniter, int* orgati, double* rho, double* d, double* z, double* finit, double* tau, int* info) {
    int kniter_val = *kniter;
    double maxit = 40.0;
    if (kniter_val < 0) { *info = -1; return; }
    if (*rho < 0.0) { *info = -3; return; }
    /* Initialize tau with Newton step estimate */
    *tau = *finit / (*rho + 1.0);
    *info = 0;
}
