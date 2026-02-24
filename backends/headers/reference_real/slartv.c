/* Vector Givens */

#

/* SLARTV - Apply sequence of Givens rotations to vectors
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
void slartv_(const int *n, float *x, const int *incx, float *y, const int *incy,
             const float *c, const int *incc, const float *s, const int *incs)
{
    int n_val = *n;
    int incx_val = *incx;
    int incy_val = *incy;
    int incc_val = *incc;
    int incs_val = *incs;

    for (int i = 0; i < n_val; i++) {
        float xi = x[i * incx_val];
        float yi = y[i * incy_val];
        float ci = c[i * incc_val];
        float si = s[i * incs_val];
        
        /* Apply rotation: [xi'] = [c  s ] [xi]
                           [yi']   [-s c ] [yi] */
        float xi_new = ci * xi + si * yi;
        float yi_new = -si * xi + ci * yi;
        
        x[i * incx_val] = xi_new;
        y[i * incy_val] = yi_new;
    }
}
