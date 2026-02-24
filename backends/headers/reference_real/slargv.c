/* Generate Givens */

#

/* SLARGV - Generate sequence of Givens rotations
   N: Number of rotations
   X: Array of x values
   INCX: Increment for X
   Y: Array of y values
   INCY: Increment for Y
   C: Output cosine values
   INCC: Increment for C
   S: Output sine values
   INCS: Increment for S
*/
void slargv_(const int *n, float *x, const int *incx, float *y, const int *incy,
             float *c, const int *incc, float *s, const int *incs)
{
    int n_val = *n;
    int incx_val = *incx;
    int incy_val = *incy;
    int incc_val = *incc;
    int incs_val = *incs;

    for (int i = 0; i < n_val; i++) {
        float xi = x[i * incx_val];
        float yi = y[i * incy_val];
        
        float r = sqrtf(xi * xi + yi * yi);
        float ci = xi / r;
        float si = yi / r;
        
        c[i * incc_val] = ci;
        s[i * incs_val] = si;
        
        /* Update x[i] to r */
        x[i * incx_val] = r;
        /* y[i] becomes 0 */
        y[i * incy_val] = 0.0f;
    }
}
