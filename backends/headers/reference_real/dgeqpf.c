/* dgeqpf */
void dgeqp3_(const int *m, const int *n, double *A, const int *lda,
             int *jpvt, double *tau, double *work, const int *lwork, int *info);

void dgeqpf_(const int* m, const int* n, double* a, const int* lda, int* jpvt,
             double* tau, double* work, int* info)
{
    int lw = (*n > 0) ? (*n + 1) : 1;
    dgeqp3_(m, n, a, lda, jpvt, tau, work, &lw, info);
}
