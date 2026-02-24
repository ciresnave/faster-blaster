/* dlantd - Double precision diagonal matrix norm */
#
void dlantd_(char* norm, int* n, double* d) {
    double nrmval = 0.0;
    for (int i = 0; i < *n; i++) {
        double absval = fabs(d[i]);
        if (absval > nrmval) nrmval = absval;
    }
}
