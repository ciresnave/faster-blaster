/**
 * UNIVERSAL BLAS/LAPACK WRAPPER GENERATION
 * 
 * The macro-based system works for ANY BLAS/LAPACK implementation!
 * 
 * The key insight: Instead of hardcoding one operation registry,
 * we can parse header files from ANY implementation and generate
 * wrappers for whatever operations it provides.
 */

/* ============================================================================
 * ARCHITECTURE: Implementation-Agnostic Wrapper Generation
 * ============================================================================ */

/*
 * The beauty of the macro system is its ABSTRACTION:
 * 
 * Instead of:
 *   "Generate wrappers for THESE 1248 operations"
 * 
 * We can have:
 *   "For EACH operation in HEADER FILE:
 *      Generate a wrapper (if it exists)"
 * 
 * This works for ANY BLAS/LAPACK implementation:
 * - OpenBLAS (header: openblas.h)
 * - Intel MKL (header: mkl.h)
 * - LAPACK (header: lapack.h)
 * - BLIS (header: blis.h)
 * - Accelerate (header: Accelerate/Accelerate.h)
 * - cuBLAS (header: cublas.h)
 * - rocBLAS (header: rocblas.h)
 * - GSL BLAS (header: gsl_blas.h)
 * - etc.
 * 
 * The operation list is auto-generated from the ACTUAL header files
 * of each implementation, so it automatically adapts to what's available.
 */

/* ============================================================================
 * EXAMPLE: How to Make It Universal
 * ============================================================================ */

/*
 * Instead of hardcoding operation_registry.h with:
 *   OP(saxpy, void, saxpy_, ...)
 *   OP(daxpy, void, daxpy_, ...)
 * 
 * We generate operation_registry.h at compile time by:
 * 1. Parsing the target implementation's header file
 * 2. Extracting available operations
 * 3. Generating OP() calls only for operations that exist
 */

/* ============================================================================
 * MULTI-BACKEND EXAMPLE: Support OpenBLAS + BLIS + Reference
 * ============================================================================ */

/*
 * CONFIG: backends/backends.h
 * 
 * #define FB_BACKEND_REFERENCE 1
 * #define FB_BACKEND_OPENBLAS  1
 * #define FB_BACKEND_BLIS      1
 * 
 * 
 * GENERATION: For each backend, include its operation registry:
 */

#ifdef FB_BACKEND_REFERENCE
    #include "backends/operations_reference.h"  /* 1248 ops extracted from faster-blaster-reference */
#endif

#ifdef FB_BACKEND_OPENBLAS
    #include "backends/operations_openblas.h"   /* N ops extracted from OpenBLAS */
#endif

#ifdef FB_BACKEND_BLIS
    #include "backends/operations_blis.h"       /* M ops extracted from BLIS */
#endif

/*
 * Each of these files is auto-generated from:
 * - Reference: ../faster-blaster-reference/src/blas_wrappers.c
 * - OpenBLAS: /usr/include/openblas.h or similar
 * - BLIS: /usr/include/blis.h or similar
 * 
 * The extraction script parses the headers and generates:
 *   OP(saxpy, void, saxpy_, (...))
 *   OP(daxpy, void, daxpy_, (...))
 *   // Only if they exist in that implementation
 */

/* ============================================================================
 * ARCHITECTURE: The Extraction Pipeline
 * ============================================================================ */

/*
 * INPUT:
 *   OpenBLAS header: /usr/include/cblas.h
 *   MKL header:      /opt/mkl/include/mkl.h
 *   Reference:       ../faster-blaster-reference/src/blas_wrappers.c
 * 
 * PROCESS (Python script):
 * 
 *   1. Parse header file
 *      ✓ Extract function declarations
 *      ✓ Parse signatures
 *      ✓ Identify parameter types
 * 
 *   2. Normalize naming conventions
 *      OpenBLAS uses: cblas_saxpy (C convention)
 *      Reference uses: saxpy_ (Fortran convention)
 *      MKL uses: cblas_saxpy (C convention)
 *      → Normalize to common format
 * 
 *   3. Generate operation registry
 *      Output: operations_<backend>.h
 *      Contains only operations that exist in that implementation
 * 
 *   4. Generate backend wrapper
 *      Output: <backend>_wrapper.c
 *      Includes registration, scoring, vtable population
 * 
 * OUTPUT:
 *   backends/operations_openblas.h
 *   backends/operations_mkl.h
 *   backends/operations_reference.h
 *   backends/wrappers_openblas.c
 *   backends/wrappers_mkl.c
 *   backends/wrappers_reference.c
 */

/* ============================================================================
 * REAL WORLD EXAMPLE: OpenBLAS Wrapper
 * ============================================================================ */

/*
 * Script: extract_from_openblas.py
 * 
 * Input:  /usr/include/cblas.h
 * Output: backends/operations_openblas.h
 * 
 * The script:
 * 1. Parses OpenBLAS header
 * 2. Finds operation declarations like:
 *    float cblas_sdot(const int N, const float *X, const int incX,
 *                     const float *Y, const int incY);
 * 3. Generates:
 *    OP(sdot, float, cblas_sdot, (int N, const float* X, int incX, const float* Y, int incY))
 *    OP(ddot, double, cblas_ddot, (...))
 *    // Only operations that are actually declared in cblas.h
 * 
 * Then in openblas_wrapper.c:
 * 
 *    #define REGISTER_OPENBLAS
 *    #define OP(name, ret, fortran_name, params) \
 *        typedef ret (*fblas_##name##_t)params;
 *    #include "operations_openblas.h"
 *    #undef OP
 *    
 *    // Now all typedefs for OpenBLAS operations are defined
 *    // Can be a different set than faster-blaster-reference!
 */

/* ============================================================================
 * FEATURE: Operation Intersection & Union
 * ============================================================================ */

/*
 * With multiple backends, we can do smart things:
 * 
 * INTERSECTION: Operations available in ALL backends
 *   Use these for correctness testing (all backends support them)
 * 
 * UNION: Operations available in ANY backend
 *   Generate wrappers for all (different per backend)
 * 
 * DIFFERENCE: Operations in one backend but not others
 *   Backend-specific extensions
 * 
 * EXAMPLE:
 * 
 *   Reference:  1248 operations
 *   OpenBLAS:   ~300 operations
 *   MKL:        ~400 operations
 *   BLIS:       ~200 operations
 * 
 *   INTERSECTION: Maybe 150 operations (common subset)
 *   → Use for correctness validation
 * 
 *   UNION: ~1500 operations total (each backend fills gaps)
 *   → All available to applications
 */

/* ============================================================================
 * IMPLEMENTATION: Flexible Backend System
 * ============================================================================ */

/*
 * backends/backend_factory.h:
 * 
 * The key is making the wrapper generation CONFIGURABLE:
 */

#ifndef FB_BACKEND_HEADER
    #error "Must define FB_BACKEND_HEADER to point to backend's header"
#endif

#ifndef FB_BACKEND_NAME
    #error "Must define FB_BACKEND_NAME (e.g., 'openblas', 'mkl', 'blis')"
#endif

/* Generate typedefs for this backend's operations */
#define OP(name, ret, fort, params) \
    typedef ret (*fblas_##name##_t)params;

/* Include this backend's operation registry (auto-generated from its headers) */
#define FB_EXPAND_OPERATIONS
#include FB_BACKEND_HEADER
#undef FB_EXPAND_OPERATIONS

#undef OP

/* Generate wrappers for this backend's operations */
#define OP(name, ret, fort, params) \
    static ret fb_##FB_BACKEND_NAME##_##name(params) { \
        static fblas_##name##_t fn = NULL; \
        if (!fn) { \
            fn = (fblas_##name##_t)FB_GET_PROC_ADDRESS(g_handle, #fort); \
        } \
        /* Convert and call */ \
    }

#define FB_EXPAND_OPERATIONS
#include FB_BACKEND_HEADER
#undef FB_EXPAND_OPERATIONS

#undef OP

/*
 * Usage:
 * 
 * To create OpenBLAS wrapper:
 *   #define FB_BACKEND_NAME "openblas"
 *   #define FB_BACKEND_HEADER "operations_openblas.h"
 *   #include "backend_factory.h"
 * 
 * To create MKL wrapper:
 *   #define FB_BACKEND_NAME "mkl"
 *   #define FB_BACKEND_HEADER "operations_mkl.h"
 *   #include "backend_factory.h"
 * 
 * To create Reference wrapper:
 *   #define FB_BACKEND_NAME "reference"
 *   #define FB_BACKEND_HEADER "operations_reference.h"
 *   #include "backend_factory.h"
 * 
 * Each produces different code, but all using the same factory!
 */

/* ============================================================================
 * EXTRACTION SCRIPT: Universal Operation Extractor
 * ============================================================================ */

/*
 * Script: extract_operations_from_any_header.py
 * 
 * Usage:
 *   python extract_operations_from_any_header.py \
 *       --header /usr/include/cblas.h \
 *       --backend openblas \
 *       --convention cblas \
 *       --output backends/operations_openblas.h
 * 
 * Supports different calling conventions:
 *   --convention cblas       # C-based (cblas_saxpy)
 *   --convention fortran     # Fortran-based (saxpy_)
 *   --convention lapack      # LAPACK-specific
 * 
 * Normalizes everything to:
 *   OP(saxpy, void, actual_function_name, (params))
 */

/* ============================================================================
 * ADVANTAGES: Why This Is Powerful
 * ============================================================================ */

/*
 * 1. AUTOMATIC ADAPTATION
 *    Update OpenBLAS version → re-extract → automatically get new operations
 * 
 * 2. MULTIPLE BACKENDS AT ONCE
 *    Support OpenBLAS AND MKL AND Reference in same build
 *    Each gets its own set of operations
 * 
 * 3. NO HANDCODING
 *    Don't manually maintain operation lists for each backend
 *    Parse from actual headers → always in sync
 * 
 * 4. EXTENSION-READY
 *    New backend? Just extract its headers
 *    Uses same macro factory → consistent interface
 * 
 * 5. CORRECTNESS VALIDATION
 *    Intersection of all backends = guaranteed operations for testing
 *    Different backends can have different operation sets
 * 
 * 6. MAINTAINABILITY
 *    Single macro factory generates all backends
 *    Change factory once → all backends updated
 */

/* ============================================================================
 * REAL EXAMPLE: Complete Multi-Backend Setup
 * ============================================================================ */

/*
 * Directory structure:
 * 
 * src/backends/
 * ├── backend_factory.h                  (Generic factory)
 * ├── operations_reference.h             (Generated from faster-blaster-reference)
 * ├── operations_openblas.h              (Generated from OpenBLAS headers)
 * ├── operations_mkl.h                   (Generated from MKL headers)
 * ├── operations_blis.h                  (Generated from BLIS headers)
 * ├── reference_wrapper.c                (Include factory with reference ops)
 * ├── openblas_wrapper.c                 (Include factory with OpenBLAS ops)
 * ├── mkl_wrapper.c                      (Include factory with MKL ops)
 * └── blis_wrapper.c                     (Include factory with BLIS ops)
 * 
 * 
 * reference_wrapper.c:
 * 
 *   #define FB_BACKEND_NAME "reference"
 *   #define FB_BACKEND_HEADER "operations_reference.h"
 *   #include "backend_factory.h"
 *   
 *   bool fb_reference_init(void) { ... }
 *   fb_backend_vtable_t* fb_reference_get_vtable(void) { ... }
 * 
 * 
 * openblas_wrapper.c:
 * 
 *   #define FB_BACKEND_NAME "openblas"
 *   #define FB_BACKEND_HEADER "operations_openblas.h"
 *   #include "backend_factory.h"
 *   
 *   bool fb_openblas_init(void) { ... }
 *   fb_backend_vtable_t* fb_openblas_get_vtable(void) { ... }
 * 
 * 
 * CMakeLists.txt:
 * 
 *   # Generate operation registries from actual backend headers
 *   add_custom_command(
 *       OUTPUT backends/operations_openblas.h
 *       COMMAND python extract_operations_from_any_header.py
 *           --header /usr/include/cblas.h
 *           --backend openblas
 *       COMMENT "Extracting OpenBLAS operations"
 *   )
 *   
 *   # Each wrapper can be enabled/disabled independently
 *   if(ENABLE_REFERENCE)
 *       target_sources(faster-blaster src/backends/reference_wrapper.c)
 *   endif()
 *   
 *   if(ENABLE_OPENBLAS)
 *       target_sources(faster-blaster src/backends/openblas_wrapper.c)
 *   endif()
 */

/* ============================================================================
 * INTELLIGENCE: Smart Operation Handling
 * ============================================================================ */

/*
 * The extraction script can be smart about operations:
 * 
 * DETECTION:
 *   Parse header → identify function signature
 *   Classify by:
 *     - Operation type (BLAS Level 1/2/3, LAPACK, etc.)
 *     - Precision (float, double, complex)
 *     - Whether it exists in this backend
 * 
 * COMPATIBILITY CHECKING:
 *   Operation exists in both OpenBLAS and Reference?
 *     → Safe to use in correctness testing
 *   Operation exists only in MKL?
 *     → Vendor-specific, document as such
 * 
 * NAMING NORMALIZATION:
 *   OpenBLAS: cblas_saxpy
 *   Reference: saxpy_
 *   MKL: cblas_saxpy
 *   → Normalize to internal name "saxpy"
 *   → actual_name field stores "cblas_saxpy" or "saxpy_"
 * 
 * SIGNATURE VALIDATION:
 *   Check that extracted signature matches expected BLAS spec
 *   If different, log warning (implementation deviation)
 */

/* ============================================================================
 * ADVANCED: Conditional Operation Support
 * ============================================================================ */

/*
 * Some operations might:
 * - Exist but behave differently
 * - Have different parameter counts
 * - Have optional parameters
 * - Be deprecated
 * 
 * The macro system can handle all of these:
 * 
 * Example: GEMM_BATCH (batch GEMM)
 * 
 *   In Reference: Not implemented
 *   In MKL: Implemented (cblas_sgemm_batch)
 *   In OpenBLAS: Implemented (cblas_dgemm_batch)
 * 
 *   Solution:
 *   
 *   operations_mkl.h:
 *     OP(gemm_batch, void, cblas_sgemm_batch, (...))
 *   
 *   operations_reference.h:
 *     // Not defined
 *   
 *   At compile time:
 *     #ifdef MKL_BACKEND
 *       fb_gemm_batch = fb_mkl_gemm_batch
 *     #else
 *       fb_gemm_batch = NULL  // Not available
 *     #endif
 */
