/* slaed6 - Solve secular equation for general n */
#
void slaed6_(int* kniter, int* orgati, float* rho, float* d, float* z, float* finit, float* tau, int* info) {
    int kniter_val = *kniter;
    float maxit = 40.0f;
    if (kniter_val < 0) { *info = -1; return; }
    if (*rho < 0.0f) { *info = -3; return; }
    /* Initialize tau with Newton step estimate */
    *tau = *finit / (*rho + 1.0f);
    *info = 0;
}
