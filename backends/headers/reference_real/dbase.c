/* dbase - Double precision base representation */
#
void dbase_(double* x, int* ibase, double* fbase) {
    *ibase = 2;
    *fbase = 2.0;
    if (*x != 0.0) {
        double absval = fabs(*x);
        int exp = (int)log(absval) / (int)log(2.0);
        *fbase = absval / pow(2.0, (double)exp);
    }
}
