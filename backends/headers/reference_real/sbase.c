/* sbase - Compute base representation */
#
void sbase_(float* x, int* ibase, float* fbase) {
    *ibase = 2;
    *fbase = 2.0f;
    if (*x != 0.0f) {
        float absval = fabs(*x);
        int exp = (int)logf(absval) / (int)logf(2.0f);
        *fbase = absval / powf(2.0f, (float)exp);
    }
}
