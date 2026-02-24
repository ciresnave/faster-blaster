/* dlargv */

#

/* DLARGV - Generate sequence of Givens rotations (double precision)
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
void dlargv_(const int *n, double *x, const int *incx, double *y, const int *incy,
             double *c, const int *incc, double *s, const int *incs)
{
    int n_val = *n;
    int incx_val = *incx;
    int incy_val = *incy;
    int incc_val = *incc;
    int incs_val = *incs;

    for (int i = 0; i < n_val; i++) {
        double xi = x[i * incx_val];
        double yi = y[i * incy_val];
        
        double r = sqrt(xi * xi + yi * yi);
        double ci = xi / r;
        double si = yi / r;
        
        c[i * incc_val] = ci;
        s[i * incs_val] = si;
        
        /* Update x[i] to r */
        x[i * incx_val] = r;
        /* y[i] becomes 0 */
        y[i * incy_val] = 0.0;
    }
}
