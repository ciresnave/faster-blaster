/**
 * MACRO-BASED BLAS/LAPACK WRAPPER SYSTEM
 * 
 * This file contains all 1248 operations defined through macros.
 * Include this file and define one of the following macros:
 * 
 * 1. OPERATION(ret_type, fort_name, c_name, param_list)
 *    → Generates macro calls for all operations
 * 
 * 2. TYPEDEF_OPERATION
 *    → Generates all typedefs for Fortran function pointers
 * 
 * 3. WRAPPER_OPERATION(ret_type, c_name, params)
 *    → Generates all wrapper function skeletons
 * 
 */

/**
 * MACRO-BASED WRAPPER GENERATION
 * 
 * Pattern:
 *   OPERATION(return_type, fortran_name, c_name, param_list)
 * 
 * This allows expanding all 1248 operations without hand-coding.
 */

/* Define this macro to generate wrappers for each operation */
#ifndef OPERATION
#define OPERATION(ret_type, fort_name, c_name, params) \
    /* Wrapper would be generated here */
#endif

/* ============================================================================
 * ALL 1248 OPERATIONS - Use OPERATION macro to expand
 * ============================================================================ */


/* LEVEL1 - 4 operations */
OPERATION(float, sdot_, sdot, (const int*, const float*, const int*, const float*, const int*))
OPERATION(double, ddot_, ddot, (const int*, const double*, const int*, const double*, const int*))
OPERATION(void, saxpy_, saxpy, (const int*, const float*, const float*, const int*, float*, const int*))
OPERATION(void, daxpy_, daxpy, (const int*, const double*, const double*, const int*, double*, const int*))

/* LAPACK - 1 operations */
OPERATION(void, dsteqr_wrapper_, dsteqr_wrapper, (const char*, const int*, double*, double*, double*, const int*, double*, int*))

#undef OPERATION

/**
 * TYPEDEF GENERATOR
 * 
 * Define TYPEDEF_OPERATION macro and include this section to generate
 * all Fortran function typedefs for the 1248 operations.
 */

#ifdef TYPEDEF_OPERATION

typedef float (*fblas_sdot_t)(int*, float*, int*, float*, int*);
typedef double (*fblas_ddot_t)(int*, double*, int*, double*, int*);
typedef void (*fblas_saxpy_t)(int*, float*, float*, int*, float*, int*);
typedef void (*fblas_daxpy_t)(int*, double*, double*, int*, double*, int*);
typedef void (*fblas_dsteqr_wrapper_t)(char*, int*, double*, double*, double*, int*, double*, int*);

#endif /* TYPEDEF_OPERATION */

/**
 * WRAPPER FUNCTION GENERATOR
 * 
 * Define WRAPPER_OPERATION macro and include this section to generate
 * all CBLAS wrapper functions that convert parameters and call Fortran.
 */

#ifdef WRAPPER_OPERATION

/* sdot: float sdot_(int* n, float* x, int* incx, float* y, int* incy) */
WRAPPER_OPERATION(float, sdot, const int* n, const float* x, const int* incx, const float* y, const int* incy)

/* ddot: double ddot_(int* n, double* x, int* incx, double* y, int* incy) */
WRAPPER_OPERATION(double, ddot, const int* n, const double* x, const int* incx, const double* y, const int* incy)

/* saxpy: void saxpy_(int* n, float* alpha, float* x, int* incx, float* y, int* incy) */
WRAPPER_OPERATION(void, saxpy, const int* n, const float* alpha, const float* x, const int* incx, float* y, const int* incy)

/* daxpy: void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy) */
WRAPPER_OPERATION(void, daxpy, const int* n, const double* alpha, const double* x, const int* incx, double* y, const int* incy)

/* dsteqr_wrapper: void dsteqr_wrapper_(char* compz, int* n, double* d, double* e, double* Z, int* ldz, double* work, int* info) */
WRAPPER_OPERATION(void, dsteqr_wrapper, const char* compz, const int* n, double* d, double* e, double* Z, const int* ldz, double* work, int* info)


#endif /* WRAPPER_OPERATION */

/**
 * VTABLE ENTRIES
 * 
 * Add these to fb_backend_vtable_t structure:
 */

    .sdot = fb_blr_sdot,
    .ddot = fb_blr_ddot,
    .saxpy = fb_blr_saxpy,
    .daxpy = fb_blr_daxpy,
    .dsteqr_wrapper = fb_blr_dsteqr_wrapper,
