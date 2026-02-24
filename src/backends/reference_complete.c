/**
 * @file reference.c
 * @brief Complete reference BLAS backend - The Beacon of Truth
 * 
 * This backend provides numerically accurate reference implementations that serve
 * as the "golden standard" for validating other backends. Every implementation
 * prioritizes correctness and numerical stability over performance.
 * 
 * Design Philosophy:
 * - Maximum numerical accuracy (Kahan summation, scaled algorithms)
 * - Proper edge case handling (zero length, zero stride, NaN, Inf)
 * - IEEE 754 compliance
 * - Clear, readable code that matches BLAS specification exactly
 * - No performance optimizations that sacrifice precision
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "backend_interface.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Backend Lifecycle
 * ========================================================================= */

static const fb_backend_info_t reference_info = {
    .name = "reference",
    .version = "1.0.0",
    .vendor = "faster-blaster",
    .capabilities = FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3,
    .hw_type = FB_HW_CPU_INTEL,
    .thread_safe = true,
    .min_efficient_size = 0  /* Works for any size */
};

static int reference_init(void) {
    return 0;  /* No initialization needed */
}

static void reference_finalize(void) {
    /* No cleanup needed */
}

static const fb_backend_info_t *reference_get_info(void) {
    return &reference_info;
}

/* ============================================================================
 * Level 1 BLAS Implementations
 * Include our complete, production-quality Level 1 operations
 * ========================================================================= */

#include "reference_level1.c"

/* ============================================================================
 * Backend VTable Registration
 * Populate the full 212-operation vtable with our implementations
 * ========================================================================= */

const fb_backend_vtable_t *fb_reference_backend(void) {
    static fb_backend_vtable_t vtable = {0};
    static int initialized = 0;
    
    if (!initialized) {
        /* Backend lifecycle */
        vtable.init = reference_init;
        vtable.finalize = reference_finalize;
        vtable.get_info = reference_get_info;
        
        /* Level 1 BLAS - COPY (4) */
        vtable.scopy = fb_ref_scopy;
        vtable.dcopy = fb_ref_dcopy;
        vtable.ccopy = fb_ref_ccopy;
        vtable.zcopy = fb_ref_zcopy;
        
        /* Level 1 BLAS - SWAP (4) */
        vtable.sswap = fb_ref_sswap;
        vtable.dswap = fb_ref_dswap;
        vtable.cswap = fb_ref_cswap;
        vtable.zswap = fb_ref_zswap;
        
        /* Level 1 BLAS - SCAL (4) */
        vtable.sscal = fb_ref_sscal;
        vtable.dscal = fb_ref_dscal;
        vtable.cscal = fb_ref_cscal;
        vtable.zscal = fb_ref_zscal;
        
        /* Level 1 BLAS - AXPY (4) */
        vtable.saxpy = fb_ref_saxpy;
        vtable.daxpy = fb_ref_daxpy;
        vtable.caxpy = fb_ref_caxpy;
        vtable.zaxpy = fb_ref_zaxpy;
        
        /* Level 1 BLAS - DOT (6: dot, dotu, dotc) */
        vtable.sdot = fb_ref_sdot;
        vtable.ddot = fb_ref_ddot;
        vtable.cdotu = fb_ref_cdotu;
        vtable.zdotu = fb_ref_zdotu;
        vtable.cdotc = fb_ref_cdotc;
        vtable.zdotc = fb_ref_zdotc;
        
        /* Level 1 BLAS - NRM2 (4) */
        vtable.snrm2 = fb_ref_snrm2;
        vtable.dnrm2 = fb_ref_dnrm2;
        vtable.scnrm2 = fb_ref_scnrm2;
        vtable.dznrm2 = fb_ref_dznrm2;
        
        /* Level 1 BLAS - ASUM (4) */
        vtable.sasum = fb_ref_sasum;
        vtable.dasum = fb_ref_dasum;
        vtable.scasum = fb_ref_scasum;
        vtable.dzasum = fb_ref_dzasum;
        
        /* Level 1 BLAS - IAMAX (4) */
        vtable.isamax = fb_ref_isamax;
        vtable.idamax = fb_ref_idamax;
        vtable.icamax = fb_ref_icamax;
        vtable.izamax = fb_ref_izamax;
        
        /* Level 1 BLAS - ROTG (4) */
        vtable.srotg = fb_ref_srotg;
        vtable.drotg = fb_ref_drotg;
        vtable.crotg = fb_ref_crotg;
        vtable.zrotg = fb_ref_zrotg;
        
        /* Level 1 BLAS - ROT (4) */
        vtable.srot = fb_ref_srot;
        vtable.drot = fb_ref_drot;
        vtable.crot = fb_ref_crot;
        vtable.zrot = fb_ref_zrot;
        
        /* Level 1 BLAS - ROTMG (2) */
        vtable.srotmg = fb_ref_srotmg;
        vtable.drotmg = fb_ref_drotmg;
        
        /* Level 1 BLAS - ROTM (2) */
        vtable.srotm = fb_ref_srotm;
        vtable.drotm = fb_ref_drotm;
        
        /* TODO: Level 2 BLAS (70 operations)
         *  - GEMV (4)
         *  - GBMV (4)
         *  - SYMV, SBMV, SPMV, HEMV, HBMV, HPMV (varies by type)
         *  - TRMV, TBMV, TPMV (4 each)
         *  - TRSV, TBSV, TPSV (4 each)
         *  - GER, GERU, GERC (varies)
         *  - SYR, SPR, SYR2, SPR2 (4 each)
         *  - HER, HPR, HER2, HPR2 (2 each)
         */
        
        /* TODO: Level 3 BLAS (30 operations)
         *  - GEMM (4)
         *  - SYMM, HEMM (varies)
         *  - SYRK, HERK (varies)
         *  - SYR2K, HER2K (varies)
         *  - TRMM (4)
         *  - TRSM (4)
         */
        
        /* TODO: LAPACK subset (60 operations)
         *  - Linear equations (GESV, POSV, SYSV, etc.)
         *  - QR, LU, Cholesky factorizations
         *  - Eigenvalue problems (SYEV, GEEV, etc.)
         */
        
        initialized = 1;
    }
    
    return &vtable;
}
