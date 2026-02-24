/* sgeqpf */
void sgeqp3_(const int *m, const int *n, float *A, const int *lda,
             int *jpvt, float *tau, float *work, const int *lwork, int *info);

void sgeqpf_(const int* m, const int* n, float* a, const int* lda, int* jpvt,
             float* tau, float* work, int* info)
{
    int lw = (*n > 0) ? (*n + 1) : 1;
    sgeqp3_(m, n, a, lda, jpvt, tau, work, &lw, info);
}
