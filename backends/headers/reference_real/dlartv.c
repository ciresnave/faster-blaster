/* Vector Givens */

#

/* DLARTV - Apply sequence of Givens rotations to vectors (double precision)
   N: Number of rotations
   X: First vector
   INCX: Increment for X
   Y: Second vector
   INCY: Increment for Y
   C: Cosine values
   INCC: Increment for C
   S: Sine values
   INCS: Increment for S
*/
void dlartv_(const int *n, double *x, const int *incx, double *y, const int *incy,
             const double *c, const int *incc, const double *s, const int *incs)
{
    int n_val = *n;
    int incx_val = *incx;
    int incy_val = *incy;
    int incc_val = *incc;
    int incs_val = *incs;

    for (int i = 0; i < n_val; i++) {
        double xi = x[i * incx_val];
        double yi = y[i * incy_val];
        double ci = c[i * incc_val];
        double si = s[i * incs_val];
        
        /* Apply rotation: [xi'] = [c  s ] [xi]
                           [yi']   [-s c ] [yi] */
        double xi_new = ci * xi + si * yi;
        double yi_new = -si * xi + ci * yi;
        
        x[i * incx_val] = xi_new;
        y[i * incy_val] = yi_new;
    }
}
