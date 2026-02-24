/* sgrowf - Growth factor calculation float */
#
void sgrowf_(int* n, float* x, int* incx, float* y, int* incy, float* growth) {
    if (*n <= 0) { *growth = 1.0f; return; }
    float norm_x = 0.0f, norm_y = 0.0f;
    for (int i = 0; i < *n; i++) {
        float vx = fabs(x[i*(*incx)]);
        float vy = fabs(y[i*(*incy)]);
        if (vx > norm_x) norm_x = vx;
        if (vy > norm_y) norm_y = vy;
    }
    *growth = (norm_x > 0.0f) ? norm_y / norm_x : 1.0f;
}
