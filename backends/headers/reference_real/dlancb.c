/* dlancb - Double precision banded matrix norm */
#
void dlancb_(char* norm, int* n, int* kl, int* ku, double* ab, int* ldab, double* work) {
    double nrmval = 0.0;
    int n_val = *n, kl_val = *kl, ku_val = *ku, ldab_val = *ldab;
    for (int j = 0; j < n_val; j++) {
        int i_min = (kl_val > j) ? 0 : j - kl_val;
        int i_max = (j + ku_val >= n_val) ? n_val - 1 : j + ku_val;
        for (int i = i_min; i <= i_max; i++) {
            double absval = fabs(ab[i - j + ku_val + j*ldab_val]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
