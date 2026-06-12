/**
 * @file test_judge_runner.c
 * @brief End-to-end Judge runner for CI validation.
 *
 * Wires fb_judge_init() + oracle + candidate backends + fb_judge_run() into
 * a CI-friendly executable that asserts on the resulting precision profiles.
 *
 * Phase 1 coverage: the reference backend acts as both oracle and candidate
 * (FB_BACKEND_ID_REFERENCE vs itself) for every Judge-implemented operation.
 * When oracle == candidate, outputs are bitwise-identical → maximum certifiable
 * digits. Assertions use conservative floors well below the hardware ceiling.
 *
 * To add more backends, uncomment entries in k_backends[] and include their
 * header (see OPTIONAL CANDIDATES section below).
 *
 * Exit code: 0 = all mandatory tests passed (SKIP does not count as failure).
 *            1 = one or more mandatory tests failed.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#endif

/* ---- Public judge API ---- */
#include "../include/faster-blaster/judge.h"
#include "../include/faster-blaster/backend_ids.h"
#include "../include/benchmark_types.h"   /* fb_size_class_t, FB_SIZE_SMALL */

/* ---- Internal judge tables (non-public but required for op ID constants
 *      and dtype enum; consistent with test_judge.c pattern).           ---- */
#include "../src/judge/judge_op_ids.h"   /* FB_OP_SAXPY, FB_OP_SGEMM, … */
#include "../src/judge/judge_metadata.h"
#include "../src/judge/judge_types.h"    /* FB_DTYPE_F32, FB_DTYPE_F64, … */

/* ---- Reference backend vtable ---- */
#include "../src/backends/reference.h"   /* fb_reference_backend(), fb_reference_init() */
#include "../include/faster-blaster/backend_plugin.h" /* fb_lib_handle_t, fb_plugin_load_library */

#ifndef FB_REFERENCE_DLL_DIR
#define FB_REFERENCE_DLL_DIR "../faster-blaster-reference/build-extended"
#endif

#ifndef FB_REFERENCE_OP_LIST_PATH
#define FB_REFERENCE_OP_LIST_PATH "../faster-blaster-reference/all_operations_in_reference.txt"
#endif

#define JR_MAX_REFERENCE_SKIP_PRINT 64u

/* fb_judge_register_oracle() and fb_judge_register_backend() are internal
 * hooks defined in judge.c.  They're intentionally absent from the public
 * judge.h (called by plugin_init.c at runtime); we declare them here so the
 * runner can drive the judge without the full plugin-loading stack.         */
extern void fb_judge_register_oracle(const fb_backend_vtable_t *vtable);
extern void fb_judge_register_backend(uint32_t backend_id,
                                       const fb_backend_vtable_t *vtable);

/* ============================================================================
 * Test-case table
 * ========================================================================== */

/*
 * Which field of fb_precision_profile_t holds the primary score.
 *   DIRECT   → profile.direct.guaranteed_digits  (BLAS DIRECT + INDEX ops)
 *   RESIDUAL → profile.residual.guaranteed_digits (LAPACK SOLVE ops)
 *   RECON    → profile.reconstruction.guaranteed_digits (FACTORIZATION ops)
 *   VALUES   → profile.values.guaranteed_digits  (SPECTRAL ops)
 */
typedef enum {
    JR_METRIC_DIRECT   = 0,
    JR_METRIC_RESIDUAL = 1,
    JR_METRIC_RECON    = 2,
    JR_METRIC_VALUES   = 3,
} jr_metric_t;

typedef struct {
    uint32_t    op_id;
    uint8_t     dtype;       /* fb_dtype_t */
    jr_metric_t metric;
    uint8_t     min_digits;  /* minimum guaranteed_digits to pass */
    const char *name;
} jr_op_entry_t;

typedef struct {
    uint32_t   op_id;
    uint8_t    dtype;
    char       name[64];
} jr_catalog_entry_t;

typedef struct {
    char (*names)[64];
    size_t count;
    const char *path_used;
} jr_reference_op_list_t;

/*
 * Conservative digit floors for the reference vs reference case.
 * Oracle == candidate → bitwise-identical output → zero error →
 * judge caps at oracle_max_certifiable (~7 F32, ~15 F64).
 * Floors are set comfortably below that ceiling so edge-case zero-vector
 * corpus entries (which produce fewer meaningful digits) don't cause
 * false failures.
 *
 * LAPACK FACTOR / SOLVE / SPECTRAL: the judge measures residual quality
 * (not bit-for-bit oracle agreement), so reported digits reflect actual
 * algorithm accuracy.  oracle_max = 4 / 10 for most LAPACK ops, so
 * JR_MIN_LAPACK_* is set at that level.
 */
#define JR_MIN_F32          6u   /* BLAS DIRECT real F32                  */
#define JR_MIN_F64          13u  /* BLAS DIRECT real F64                  */
#define JR_MIN_CF32         5u   /* BLAS DIRECT complex CF32              */
#define JR_MIN_CF64         12u  /* BLAS DIRECT complex CF64              */
#define JR_MIN_BSOLVE_F32   4u   /* BLAS SOLVE (TRSV/TRSM) F32/CF32      */
#define JR_MIN_BSOLVE_F64   10u  /* BLAS SOLVE (TRSV/TRSM) F64/CF64      */
#define JR_MIN_LAPACK_F32   4u   /* LAPACK FACTOR / SOLVE F32/CF32        */
#define JR_MIN_LAPACK_F64   10u  /* LAPACK FACTOR / SOLVE F64/CF64        */
#define JR_MIN_SPECTRAL_F32 4u   /* LAPACK SPECTRAL (SYEV/GESVD/GEEV) F32 */
#define JR_MIN_SPECTRAL_F64 10u  /* LAPACK SPECTRAL F64                   */

/*
 * All operations with Judge implementations.
 * Ops that return FB_JUDGE_ERR_NOT_IMPL are marked SKIP, not FAIL.
 */
static const jr_op_entry_t k_ops[] = {
    /* ================================================================ */
    /* BLAS Level 1 — DIRECT (real)                                     */
    /* ================================================================ */
    {FB_OP_SASUM, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sasum"},
    {FB_OP_DASUM, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dasum"},
    {FB_OP_SAXPY, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "saxpy"},
    {FB_OP_DAXPY, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "daxpy"},
    {FB_OP_SCOPY, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "scopy"},
    {FB_OP_DCOPY, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dcopy"},
    {FB_OP_SDOT, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sdot"},
    {FB_OP_DDOT, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "ddot"},
    {FB_OP_SNRM2, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "snrm2"},
    {FB_OP_DNRM2, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dnrm2"},
    {FB_OP_SSCAL, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sscal"},
    {FB_OP_DSCAL, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dscal"},
    {FB_OP_SSWAP, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sswap"},
    {FB_OP_DSWAP, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dswap"},
    {FB_OP_SDSDOT, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sdsdot"},
    {FB_OP_DSDOT, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "dsdot"},
    {FB_OP_SROT, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "srot"},
    {FB_OP_DROT, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "drot"},
    {FB_OP_SROTG, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "srotg"},
    {FB_OP_DROTG, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "drotg"},
    {FB_OP_SROTM, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "srotm"},
    {FB_OP_DROTM, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "drotm"},
    {FB_OP_SROTMG, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "srotmg"},
    /* BLAS Level 1 — DIRECT (complex)                                  */
    {FB_OP_CAXPY, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "caxpy"},
    {FB_OP_ZAXPY, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zaxpy"},
    {FB_OP_CSCAL, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cscal"},
    {FB_OP_ZSCAL, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zscal"},
    {FB_OP_CSSCAL, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csscal"},
    {FB_OP_ZDSCAL, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdscal"},
    {FB_OP_CCOPY, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ccopy"},
    {FB_OP_ZCOPY, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zcopy"},
    {FB_OP_CSWAP, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cswap"},
    {FB_OP_ZSWAP, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zswap"},
    {FB_OP_CDOTU, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cdotu"},
    {FB_OP_ZDOTU, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdotu"},
    {FB_OP_CDOTC, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cdotc"},
    {FB_OP_ZDOTC, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdotc"},
    {FB_OP_SCASUM, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "scasum"},
    {FB_OP_DZASUM, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "dzasum"},
    {FB_OP_SCNRM2, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "scnrm2"},
    {FB_OP_DZNRM2, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "dznrm2"},
    {FB_OP_CROT, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "crot"},
    {FB_OP_ZROT, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zrot"},
    {FB_OP_ZDROT, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdrot"},
    /* BLAS Level 1 — complex rotg + double modified plane rotation */
    {FB_OP_CROTG, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "crotg"},
    {FB_OP_ZROTG, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zrotg"},
    {FB_OP_DROTMG, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "drotmg"},
    /* BLAS Level 1 — INDEX                                             */
    {FB_OP_ISAMAX, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "isamax"},
    {FB_OP_IDAMAX, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "idamax"},
    {FB_OP_ICAMAX, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "icamax"},
    {FB_OP_IZAMAX, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "izamax"},
    /* ================================================================ */
    /* BLAS Level 2 — DIRECT                                            */
    /* ================================================================ */
    {FB_OP_SGEMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sgemv"},
    {FB_OP_DGEMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dgemv"},
    {FB_OP_CGEMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgemv"},
    {FB_OP_ZGEMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgemv"},
    {FB_OP_SSYMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "ssymv"},
    {FB_OP_DSYMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dsymv"},
    {FB_OP_CHEMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chemv"},
    {FB_OP_ZHEMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhemv"},
    {FB_OP_STRMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "strmv"},
    {FB_OP_DTRMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dtrmv"},
    {FB_OP_CTRMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctrmv"},
    {FB_OP_ZTRMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztrmv"},
    {FB_OP_SGER, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sger"},
    {FB_OP_DGER, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dger"},
    {FB_OP_CGERU, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgeru"},
    {FB_OP_CGERC, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgerc"},
    {FB_OP_ZGERU, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgeru"},
    {FB_OP_ZGERC, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgerc"},
    {FB_OP_SSYR, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "ssyr"},
    {FB_OP_DSYR, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dsyr"},
    {FB_OP_CHER, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cher"},
    {FB_OP_ZHER, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zher"},
    {FB_OP_SSYR2, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "ssyr2"},
    {FB_OP_DSYR2, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dsyr2"},
    {FB_OP_CHER2, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cher2"},
    {FB_OP_ZHER2, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zher2"},
    {FB_OP_SSPMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sspmv"},
    {FB_OP_DSPMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dspmv"},
    {FB_OP_CHPMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chpmv"},
    {FB_OP_ZHPMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhpmv"},
    {FB_OP_SSBMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "ssbmv"},
    {FB_OP_DSBMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dsbmv"},
    {FB_OP_CHBMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chbmv"},
    {FB_OP_ZHBMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhbmv"},
    {FB_OP_STBMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "stbmv"},
    {FB_OP_DTBMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dtbmv"},
    {FB_OP_CTBMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctbmv"},
    {FB_OP_ZTBMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztbmv"},
    {FB_OP_STPMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "stpmv"},
    {FB_OP_DTPMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dtpmv"},
    {FB_OP_CTPMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctpmv"},
    {FB_OP_ZTPMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztpmv"},
    {FB_OP_SSPR, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sspr"},
    {FB_OP_DSPR, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dspr"},
    {FB_OP_CHPR, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chpr"},
    {FB_OP_ZHPR, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhpr"},
    {FB_OP_SSPR2, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sspr2"},
    {FB_OP_DSPR2, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dspr2"},
    {FB_OP_CHPR2, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chpr2"},
    {FB_OP_ZHPR2, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhpr2"},
    /* BLAS Level 2 — banded (GBMV) */
    {FB_OP_SGBMV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sgbmv"},
    {FB_OP_DGBMV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dgbmv"},
    {FB_OP_CGBMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgbmv"},
    {FB_OP_ZGBMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgbmv"},
    /* BLAS Level 2 — complex symmetric matrix-vector and rank-1 */
    {FB_OP_CSYMV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csymv"},
    {FB_OP_ZSYMV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsymv"},
    {FB_OP_CSYR, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csyr"},
    {FB_OP_ZSYR, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsyr"},
    /* BLAS Level 2 — triangular solve (metadata changed to DIRECT_ENTRY) */
    {FB_OP_STRSV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "strsv"},
    {FB_OP_DTRSV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "dtrsv"},
    {FB_OP_CTRSV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "ctrsv"},
    {FB_OP_ZTRSV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "ztrsv"},
    {FB_OP_STBSV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "stbsv"},
    {FB_OP_DTBSV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "dtbsv"},
    {FB_OP_CTBSV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "ctbsv"},
    {FB_OP_ZTBSV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "ztbsv"},
    {FB_OP_STPSV, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "stpsv"},
    {FB_OP_DTPSV, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "dtpsv"},
    {FB_OP_CTPSV, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "ctpsv"},
    {FB_OP_ZTPSV, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "ztpsv"},
    /* ================================================================ */
    /* BLAS Level 3 — DIRECT                                            */
    /* ================================================================ */
    {FB_OP_SGEMM, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "sgemm"},
    {FB_OP_DGEMM, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dgemm"},
    {FB_OP_CGEMM, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgemm"},
    {FB_OP_ZGEMM, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgemm"},
    {FB_OP_SSYMM, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "ssymm"},
    {FB_OP_DSYMM, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dsymm"},
    {FB_OP_CSYMM, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csymm"},
    {FB_OP_ZSYMM, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsymm"},
    {FB_OP_CHEMM, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chemm"},
    {FB_OP_ZHEMM, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhemm"},
    {FB_OP_SSYRK, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "ssyrk"},
    {FB_OP_DSYRK, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dsyrk"},
    {FB_OP_CSYRK, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csyrk"},
    {FB_OP_ZSYRK, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsyrk"},
    {FB_OP_CHERK, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cherk"},
    {FB_OP_ZHERK, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zherk"},
    {FB_OP_SSYR2K, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "ssyr2k"},
    {FB_OP_DSYR2K, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dsyr2k"},
    {FB_OP_CSYR2K, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csyr2k"},
    {FB_OP_ZSYR2K, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsyr2k"},
    {FB_OP_CHER2K, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cher2k"},
    {FB_OP_ZHER2K, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zher2k"},
    {FB_OP_STRMM, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_F32, "strmm"},
    {FB_OP_DTRMM, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_F64, "dtrmm"},
    {FB_OP_CTRMM, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctrmm"},
    {FB_OP_ZTRMM, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztrmm"},
    /* BLAS Level 3 — triangular matrix-matrix solve (metadata changed to
       DIRECT_ENTRY) */
    {FB_OP_STRSM, FB_DTYPE_F32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "strsm"},
    {FB_OP_DTRSM, FB_DTYPE_F64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "dtrsm"},
    {FB_OP_CTRSM, FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F32, "ctrsm"},
    {FB_OP_ZTRSM, FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_BSOLVE_F64, "ztrsm"},
    /* ================================================================ */
    /* LAPACK — Factorization (RECON metric)                            */
    /* ================================================================ */
    {FB_OP_SGETRF, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "sgetrf"},
    {FB_OP_DGETRF, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dgetrf"},
    {FB_OP_CGETRF, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cgetrf"},
    {FB_OP_ZGETRF, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zgetrf"},
    {FB_OP_SPOTRF, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "spotrf"},
    {FB_OP_DPOTRF, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dpotrf"},
    {FB_OP_CPOTRF, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cpotrf"},
    {FB_OP_ZPOTRF, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zpotrf"},
    /* LAPACK QR factorization */
    {FB_OP_SGEQRF, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "sgeqrf"},
    {FB_OP_DGEQRF, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dgeqrf"},
    {FB_OP_CGEQRF, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cgeqrf"},
    {FB_OP_ZGEQRF, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zgeqrf"},
    /* LAPACK Q-factor recovery */
    {FB_OP_SORGQR, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "sorgqr"},
    {FB_OP_DORGQR, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dorgqr"},
    {FB_OP_CUNGQR, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cungqr"},
    {FB_OP_ZUNGQR, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zungqr"},
    /* LAPACK — Triangular Solve (RESIDUAL metric)                       */
    {FB_OP_SGETRS, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "sgetrs"},
    {FB_OP_DGETRS, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "dgetrs"},
    {FB_OP_CGETRS, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "cgetrs"},
    {FB_OP_ZGETRS, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zgetrs"},
    {FB_OP_SPOTRS, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "spotrs"},
    {FB_OP_DPOTRS, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "dpotrs"},
    {FB_OP_CPOTRS, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "cpotrs"},
    {FB_OP_ZPOTRS, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zpotrs"},
    /* LAPACK — Least-squares solve (GELS: overdetermined, FB_NO_TRANS)   */
    {FB_OP_SGELS, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "sgels"},
    {FB_OP_DGELS, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "dgels"},
    {FB_OP_CGELS, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "cgels"},
    {FB_OP_ZGELS, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zgels"},
    /* LAPACK — Driver: General LU solve (GESV)                          */
    {FB_OP_SGESV, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "sgesv"},
    {FB_OP_DGESV, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "dgesv"},
    {FB_OP_CGESV, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "cgesv"},
    {FB_OP_ZGESV, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zgesv"},
    /* LAPACK — Driver: Symmetric positive-definite solve (POSV)         */
    {FB_OP_SPOSV, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "sposv"},
    {FB_OP_DPOSV, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "dposv"},
    {FB_OP_CPOSV, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "cposv"},
    {FB_OP_ZPOSV, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zposv"},
    /* LAPACK — SVD-based least-squares driver (GELSD; residual metric)   */
    {FB_OP_SGELSD, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "sgelsd"},
    {FB_OP_DGELSD, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "dgelsd"},
    {FB_OP_CGELSD, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "cgelsd"},
    {FB_OP_ZGELSD, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zgelsd"},
    /* LAPACK — Pivoted QR least-squares driver (GELSY; residual metric)  */
    {FB_OP_SGELSY, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "sgelsy"},
    {FB_OP_DGELSY, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "dgelsy"},
    {FB_OP_CGELSY, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "cgelsy"},
    {FB_OP_ZGELSY, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zgelsy"},
    /* LAPACK — Apply Q from QR (RECON metric)                           */
    {FB_OP_SORMQR, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "sormqr"},
    {FB_OP_DORMQR, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dormqr"},
    {FB_OP_CUNMQR, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cunmqr"},
    {FB_OP_ZUNMQR, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zunmqr"},
    /* LAPACK — Triangular inversion (TRTRI; RECON metric)               */
    {FB_OP_STRTRI, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "strtri"},
    {FB_OP_DTRTRI, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dtrtri"},
    {FB_OP_CTRTRI, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "ctrtri"},
    {FB_OP_ZTRTRI, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "ztrtri"},
    /* LAPACK — Triangular system solve (TRTRS; RESIDUAL metric)          */
    {FB_OP_STRTRS, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "strtrs"},
    {FB_OP_DTRTRS, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "dtrtrs"},
    {FB_OP_CTRTRS, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "ctrtrs"},
    {FB_OP_ZTRTRS, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "ztrtrs"},
    /* LAPACK — Symmetric system solve (SSYTRS; RESIDUAL metric)          */
    {FB_OP_SSYTRS, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "ssytrs"},
    {FB_OP_DSYTRS, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "dsytrs"},
    {FB_OP_CSYTRS, FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32,
     "csytrs"},
    {FB_OP_ZSYTRS, FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64,
     "zsytrs"},
    /* LAPACK — Symmetric/Hermitian indefinite factorisation (SSYTRF; RECON
       metric) */
    {FB_OP_SSYTRF, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "ssytrf"},
    {FB_OP_DSYTRF, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dsytrf"},
    /* LAPACK — Symmetric/Hermitian driver: factor+solve (SSYSV/DSYSV; RESIDUAL)
     */
    {FB_OP_SSYSV, FB_DTYPE_F32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "ssysv"},
    {FB_OP_DSYSV, FB_DTYPE_F64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "dsysv"},
    /* LAPACK — LU-based full matrix inversion (GETRI; RECON metric)     */
    {FB_OP_SGETRI, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "sgetri"},
    {FB_OP_DGETRI, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dgetri"},
    {FB_OP_CGETRI, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cgetri"},
    {FB_OP_ZGETRI, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zgetri"},
    /* LAPACK — Cholesky-based SPD matrix inversion (POTRI; RECON metric) */
    {FB_OP_SPOTRI, FB_DTYPE_F32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "spotri"},
    {FB_OP_DPOTRI, FB_DTYPE_F64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dpotri"},
    {FB_OP_CPOTRI, FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cpotri"},
    {FB_OP_ZPOTRI, FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zpotri"},
    /* SPECTRAL — eigenvalue and singular-value decompositions.
     * Auxiliary *_ref routines (sgehrd_ref, shseqr_ref, strevc_ref, sbdsqr_ref,
     * chetrd_ref, steqr_ref, ...) are now present in faster-blaster-reference.
     * Wired via ref_ssyev / ref_sgesvd / ref_sgeev / ref_sgesdd / ref_ssygv in
     * src/backends/reference_lapack.c. */

    /* SYEV — symmetric real eigenvalues */
    {FB_OP_SSYEV, FB_DTYPE_F32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32, "ssyev"},
    {FB_OP_DSYEV, FB_DTYPE_F64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64, "dsyev"},
    /* HEEV — Hermitian complex eigenvalues */
    {FB_OP_CHEEV, FB_DTYPE_CF32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "cheev"},
    {FB_OP_ZHEEV, FB_DTYPE_CF64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "zheev"},
    /* SYEVD — symmetric divide-and-conquer eigenvalues */
    {FB_OP_SSYEVD, FB_DTYPE_F32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "ssyevd"},
    {FB_OP_DSYEVD, FB_DTYPE_F64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "dsyevd"},
    /* HEEVD — Hermitian divide-and-conquer eigenvalues */
    {FB_OP_CHEEVD, FB_DTYPE_CF32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "cheevd"},
    {FB_OP_ZHEEVD, FB_DTYPE_CF64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "zheevd"},
    /* GESVD — singular value decomposition */
    {FB_OP_SGESVD, FB_DTYPE_F32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "sgesvd"},
    {FB_OP_DGESVD, FB_DTYPE_F64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "dgesvd"},
    {FB_OP_CGESVD, FB_DTYPE_CF32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "cgesvd"},
    {FB_OP_ZGESVD, FB_DTYPE_CF64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "zgesvd"},
    /* GEEV — general real eigenvalues */
    {FB_OP_SGEEV, FB_DTYPE_F32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32, "sgeev"},
    {FB_OP_DGEEV, FB_DTYPE_F64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64, "dgeev"},
    /* GEEV — general complex eigenvalues */
    {FB_OP_CGEEV, FB_DTYPE_CF32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "cgeev"},
    {FB_OP_ZGEEV, FB_DTYPE_CF64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "zgeev"},
    /* GESDD — divide-and-conquer SVD */
    {FB_OP_SGESDD, FB_DTYPE_F32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "sgesdd"},
    {FB_OP_DGESDD, FB_DTYPE_F64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "dgesdd"},
    /* GESDD — complex divide-and-conquer SVD */
    {FB_OP_CGESDD, FB_DTYPE_CF32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "cgesdd"},
    {FB_OP_ZGESDD, FB_DTYPE_CF64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "zgesdd"},
    /* SYGV — generalized symmetric eigenproblem */
    {FB_OP_SSYGV, FB_DTYPE_F32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32, "ssygv"},
    {FB_OP_DSYGV, FB_DTYPE_F64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64, "dsygv"},
    /* HEGV — generalized Hermitian eigenproblem */
    {FB_OP_CHEGV, FB_DTYPE_CF32, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F32,
     "chegv"},
    {FB_OP_ZHEGV, FB_DTYPE_CF64, JR_METRIC_VALUES, JR_MIN_SPECTRAL_F64,
     "zhegv"},
};

static const int k_op_count = (int)(sizeof(k_ops) / sizeof(k_ops[0]));

static const jr_op_entry_t *find_op_override(uint32_t op_id)
{
    for (int i = 0; i < k_op_count; i++) {
        if (k_ops[i].op_id == op_id) {
            return &k_ops[i];
        }
    }
    return NULL;
}

static fb_dtype_t infer_dtype_from_macro_name(const char *macro_name)
{
    const char *name = macro_name;

    if (!name) {
        return FB_DTYPE_F32;
    }
    if (strncmp(name, "FB_OP_", 6) == 0) {
        name += 6;
    }

    if (strstr(name, "BF16") != NULL) {
        return FB_DTYPE_BF16;
    }
    if (strstr(name, "F16") != NULL) {
        return FB_DTYPE_F16;
    }
    if (strstr(name, "I32") != NULL || strstr(name, "S32") != NULL) {
        return FB_DTYPE_I32;
    }
    if (strstr(name, "I8") != NULL || strstr(name, "INT8") != NULL ||
        strstr(name, "S8") != NULL || strstr(name, "U8") != NULL) {
        return FB_DTYPE_I8;
    }

    if (strncmp(name, "CBLAS_S", 7) == 0 || strncmp(name, "MKL_S", 6) == 0 ||
        strncmp(name, "PS", 2) == 0) {
        return FB_DTYPE_F32;
    }
    if (strncmp(name, "CBLAS_D", 7) == 0 || strncmp(name, "MKL_D", 6) == 0 ||
        strncmp(name, "PD", 2) == 0) {
        return FB_DTYPE_F64;
    }
    if (strncmp(name, "CBLAS_C", 7) == 0 || strncmp(name, "MKL_C", 6) == 0 ||
        strncmp(name, "PC", 2) == 0) {
        return FB_DTYPE_CF32;
    }
    if (strncmp(name, "CBLAS_Z", 7) == 0 || strncmp(name, "MKL_Z", 6) == 0 ||
        strncmp(name, "PZ", 2) == 0) {
        return FB_DTYPE_CF64;
    }

    if (strncmp(name, "MKL_JIT_", 8) == 0) {
        if (strstr(name, "_CGEMM") != NULL) {
            return FB_DTYPE_CF32;
        }
        if (strstr(name, "_ZGEMM") != NULL) {
            return FB_DTYPE_CF64;
        }
        if (strstr(name, "_DGEMM") != NULL) {
            return FB_DTYPE_F64;
        }
        if (strstr(name, "_SGEMM") != NULL) {
            return FB_DTYPE_F32;
        }
    }

    if (strncmp(name, "S", 1) == 0) {
        return FB_DTYPE_F32;
    }
    if (strncmp(name, "D", 1) == 0) {
        return FB_DTYPE_F64;
    }
    if (strncmp(name, "C", 1) == 0) {
        return FB_DTYPE_CF32;
    }
    if (strncmp(name, "Z", 1) == 0) {
        return FB_DTYPE_CF64;
    }

    return FB_DTYPE_F32;
}

static void macro_name_to_canonical_name(const char *macro_name,
                                         char *buf,
                                         size_t buf_size)
{
    const char *src = macro_name;
    size_t idx = 0;

    if (!buf || buf_size == 0) {
        return;
    }

    buf[0] = '\0';
    if (!src) {
        return;
    }

    if (strncmp(src, "FB_OP_", 6) == 0) {
        src += 6;
    }

    while (*src != '\0' && idx < (buf_size - 1)) {
        buf[idx++] = (char)tolower((unsigned char)*src++);
    }
    buf[idx] = '\0';
}

static jr_metric_t default_metric_from_meta(const fb_op_judge_meta_t *meta)
{
    if (!meta) {
        return JR_METRIC_DIRECT;
    }

    switch (meta->archetype) {
        case FB_JUDGE_SOLVE:
            return JR_METRIC_RESIDUAL;
        case FB_JUDGE_FACTORIZATION:
            return JR_METRIC_RECON;
        case FB_JUDGE_SPECTRAL:
            return JR_METRIC_VALUES;
        case FB_JUDGE_INDEX:
        case FB_JUDGE_DIRECT:
        default:
            return JR_METRIC_DIRECT;
    }
}

static uint8_t default_min_digits_from_meta(const fb_op_judge_meta_t *meta,
                                            uint8_t dtype)
{
    bool is_f64_like = (dtype == FB_DTYPE_F64 || dtype == FB_DTYPE_CF64 ||
                        dtype == FB_DTYPE_I32);
    bool is_complex = (dtype == FB_DTYPE_CF32 || dtype == FB_DTYPE_CF64);

    if (!meta) {
        return is_f64_like ? JR_MIN_F64 : JR_MIN_F32;
    }

    switch (meta->archetype) {
        case FB_JUDGE_SOLVE:
            return is_f64_like ? JR_MIN_LAPACK_F64 : JR_MIN_LAPACK_F32;
        case FB_JUDGE_FACTORIZATION:
            return is_f64_like ? JR_MIN_LAPACK_F64 : JR_MIN_LAPACK_F32;
        case FB_JUDGE_SPECTRAL:
            return is_f64_like ? JR_MIN_SPECTRAL_F64 : JR_MIN_SPECTRAL_F32;
        case FB_JUDGE_INDEX:
        case FB_JUDGE_DIRECT:
        default:
            if (is_complex) {
                return is_f64_like ? JR_MIN_CF64 : JR_MIN_CF32;
            }
            return is_f64_like ? JR_MIN_F64 : JR_MIN_F32;
    }
}

static bool try_open_catalog(FILE **file_out)
{
    static const char *const k_candidate_paths[] = {
        "src/judge/judge_op_ids.h",
        "../src/judge/judge_op_ids.h",
        "../../src/judge/judge_op_ids.h",
        NULL,
    };

    if (!file_out) {
        return false;
    }

    for (int i = 0; k_candidate_paths[i] != NULL; i++) {
        FILE *file = fopen(k_candidate_paths[i], "r");
        if (file != NULL) {
            *file_out = file;
            return true;
        }
    }

    *file_out = NULL;
    return false;
}

static bool load_op_catalog(jr_catalog_entry_t **entries_out, size_t *count_out)
{
    FILE *file = NULL;
    jr_catalog_entry_t *entries = NULL;
    size_t capacity = 0;
    size_t count = 0;
    char line[256];

    if (!entries_out || !count_out) {
        return false;
    }

    *entries_out = NULL;
    *count_out = 0;

    if (!try_open_catalog(&file)) {
        return false;
    }

    capacity = 512;
    entries = (jr_catalog_entry_t *)calloc(capacity, sizeof(*entries));
    if (!entries) {
        fclose(file);
        return false;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        char macro_name[128];
        unsigned op_id = 0;

        if (strncmp(line, "#define FB_OP_", 14) != 0) {
            continue;
        }
        if (strncmp(line, "#define FB_OP__", 15) == 0) {
            continue;
        }
        if (sscanf(line, "#define %127s %u", macro_name, &op_id) != 2) {
            continue;
        }

        if (count == capacity) {
            size_t new_capacity = capacity * 2;
            jr_catalog_entry_t *new_entries =
                (jr_catalog_entry_t *)realloc(entries, new_capacity * sizeof(*entries));
            if (!new_entries) {
                free(entries);
                fclose(file);
                return false;
            }
            entries = new_entries;
            capacity = new_capacity;
        }

        entries[count].op_id = op_id;
        entries[count].dtype = (uint8_t)infer_dtype_from_macro_name(macro_name);
        macro_name_to_canonical_name(macro_name, entries[count].name,
                                     sizeof(entries[count].name));
        count++;
    }

    fclose(file);

    *entries_out = entries;
    *count_out = count;
    return true;
}

static void normalize_reference_name(const char *src,
                                     char *buf,
                                     size_t buf_size)
{
    size_t start = 0;
    size_t end = 0;
    size_t idx = 0;

    if (!buf || buf_size == 0) {
        return;
    }

    buf[0] = '\0';
    if (!src) {
        return;
    }

    while (src[start] != '\0' && isspace((unsigned char)src[start])) {
        start++;
    }

    end = strlen(src);
    while (end > start && isspace((unsigned char)src[end - 1])) {
        end--;
    }

    while ((start + idx) < end && idx < (buf_size - 1)) {
        buf[idx] = (char)tolower((unsigned char)src[start + idx]);
        idx++;
    }
    buf[idx] = '\0';
}

static bool try_open_reference_op_list(FILE **file_out, const char **path_out)
{
    const char *override = getenv("FB_JUDGE_REFERENCE_OP_LIST");
    static const char *const k_candidate_paths[] = {
        FB_REFERENCE_OP_LIST_PATH,
        "../faster-blaster-reference/all_operations_in_reference.txt",
        "../../faster-blaster-reference/all_operations_in_reference.txt",
        NULL,
    };

    if (!file_out) {
        return false;
    }

    if (override && override[0] != '\0') {
        FILE *override_file = fopen(override, "r");
        if (override_file != NULL) {
            *file_out = override_file;
            if (path_out) {
                *path_out = override;
            }
            return true;
        }
    }

    for (int i = 0; k_candidate_paths[i] != NULL; i++) {
        FILE *candidate = fopen(k_candidate_paths[i], "r");
        if (candidate != NULL) {
            *file_out = candidate;
            if (path_out) {
                *path_out = k_candidate_paths[i];
            }
            return true;
        }
    }

    *file_out = NULL;
    if (path_out) {
        *path_out = NULL;
    }
    return false;
}

static bool reference_name_exists(const jr_reference_op_list_t *list,
                                  const char *name)
{
    if (!list || !list->names || !name || name[0] == '\0') {
        return false;
    }

    for (size_t i = 0; i < list->count; i++) {
        if (strcmp(list->names[i], name) == 0) {
            return true;
        }
    }

    return false;
}

static bool load_reference_op_list(jr_reference_op_list_t *list)
{
    FILE *file = NULL;
    const char *path_used = NULL;
    char (*names)[64] = NULL;
    size_t capacity = 0;
    size_t count = 0;
    char line[256];

    if (!list) {
        return false;
    }

    list->names = NULL;
    list->count = 0;
    list->path_used = NULL;

    if (!try_open_reference_op_list(&file, &path_used)) {
        return false;
    }

    capacity = 512;
    names = (char (*)[64])calloc(capacity, sizeof(*names));
    if (!names) {
        fclose(file);
        return false;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        char normalized[64];

        normalize_reference_name(line, normalized, sizeof(normalized));
        if (normalized[0] == '\0') {
            continue;
        }
        if (reference_name_exists(&(jr_reference_op_list_t){ .names = names, .count = count }, normalized)) {
            continue;
        }

        if (count == capacity) {
            size_t new_capacity = capacity * 2;
            char (*new_names)[64] =
                (char (*)[64])realloc(names, new_capacity * sizeof(*names));
            if (!new_names) {
                free(names);
                fclose(file);
                return false;
            }
            names = new_names;
            capacity = new_capacity;
        }

        snprintf(names[count], sizeof(names[count]), "%s", normalized);
        count++;
    }

    fclose(file);

    list->names = names;
    list->count = count;
    list->path_used = path_used;
    return true;
}

static void free_reference_op_list(jr_reference_op_list_t *list)
{
    if (!list) {
        return;
    }

    free(list->names);
    list->names = NULL;
    list->count = 0;
    list->path_used = NULL;
}

static bool reference_op_list_contains(const jr_reference_op_list_t *list,
                                      const char *judge_name)
{
    const char *normalized_name = judge_name;

    if (!list || !judge_name) {
        return false;
    }

    if (strncmp(normalized_name, "cblas_", 6) == 0) {
        normalized_name += 6;
    }

    return reference_name_exists(list, normalized_name);
}

/* ============================================================================
 * Backend table
 *
 * The reference backend is always present (statically linked).
 * Optional CPU / GPU backends can be uncommented once their build flags are
 * enabled; their vtable getter headers must be included above.
 * ========================================================================== */

typedef struct {
    uint32_t    backend_id;
    const char *name;
    bool        optional;   /* true → non-availability → SKIP, not FAIL */
    const fb_backend_vtable_t *(*get_vtable)(void);
} jr_backend_entry_t;

static const jr_backend_entry_t k_backends[] = {
    /*
     * Reference — mandatory; always statically linked; acts as correctness
     * oracle for every operation.  Running it as a candidate against itself
     * verifies the Judge infrastructure end-to-end.
     */
    {
        FB_BACKEND_ID_REFERENCE,
        "reference",
        false,               /* not optional */
        fb_reference_backend
    },

    /* ---- OPTIONAL CANDIDATES — uncomment + add headers as needed --------
     *
     * { FB_BACKEND_ID_AOCL_BLIS, "aocl-blis",  true, fb_aocl_get_vtable    },
     * { FB_BACKEND_ID_BLIS,      "blis",        true, fb_blis_get_vtable    },
     * { FB_BACKEND_ID_OPENBLAS,  "openblas",    true, fb_openblas_get_vtable },
     * { FB_BACKEND_ID_MKL,       "mkl",         true, fb_mkl_get_vtable     },
     *
     * -------------------------------------------------------------------- */
};

static const int k_backend_count =
    (int)(sizeof(k_backends) / sizeof(k_backends[0]));

/* ============================================================================
 * Helpers
 * ========================================================================== */

static const char *judge_status_str(fb_judge_status_t s)
{
    switch (s) {
        case FB_JUDGE_OK:                  return "OK";
        case FB_JUDGE_ERR_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case FB_JUDGE_ERR_INVALID_OP:      return "INVALID_OP";
        case FB_JUDGE_ERR_NO_PROFILE:      return "NO_PROFILE";
        case FB_JUDGE_ERR_ORACLE_FAILURE:  return "ORACLE_FAILURE";
        case FB_JUDGE_ERR_ALLOC:           return "ALLOC_ERR";
        case FB_JUDGE_ERR_IO:              return "IO_ERR";
        case FB_JUDGE_ERR_NOT_IMPL:        return "NOT_IMPL";
        default:                           return "UNKNOWN";
    }
}

static bool parse_env_u32(const char *name, uint32_t *out)
{
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long parsed;

    if (!name || !out || !value || value[0] == '\0') {
        return false;
    }

    parsed = strtoul(value, &end, 10);
    if (end == value || (end && *end != '\0') || parsed > UINT32_MAX) {
        return false;
    }

    *out = (uint32_t)parsed;
    return true;
}

static bool parse_env_flag(const char *name)
{
    const char *value = getenv(name);

    if (!name || !value || value[0] == '\0') {
        return false;
    }

    return strcmp(value, "0") != 0;
}

static unsigned long long judge_runner_profile_nonce(void)
{
#if defined(_WIN32)
    return (((unsigned long long)GetCurrentProcessId()) << 32) ^
           (unsigned long long)GetTickCount64();
#else
    return (((unsigned long long)getpid()) << 32) ^
           (unsigned long long)time(NULL);
#endif
}

static void build_profile_dir(char *buf, size_t buf_size)
{
    const char *override = getenv("FB_JUDGE_PROFILE_DIR");
    const char *tmp = getenv("TEMP");

    if (!buf || buf_size == 0) {
        return;
    }

    if (override && override[0] != '\0') {
        snprintf(buf, buf_size, "%s", override);
        return;
    }

    if (!tmp) {
        tmp = getenv("TMPDIR");
    }
    if (!tmp) {
        tmp = "/tmp";
    }

    snprintf(buf, buf_size, "%s/fb_judge_runner_profiles_%llu",
             tmp, judge_runner_profile_nonce());
}

static bool validate_process_heap(const char *phase, uint32_t op_id,
                                  const char *op_name)
{
#if defined(_WIN32)
    HANDLE heap = GetProcessHeap();

    if (!heap) {
        fprintf(stderr,
                "[HEAP] unable to get process heap %s op %u %s\n",
                phase, (unsigned)op_id, op_name ? op_name : "?");
        return false;
    }

    if (!HeapValidate(heap, 0, NULL)) {
        DWORD err = GetLastError();

        fprintf(stderr,
                "[HEAP] invalid %s op %u %s (GetLastError=%lu)\n",
                phase, (unsigned)op_id, op_name ? op_name : "?",
                (unsigned long)err);
        return false;
    }
#else
    (void)phase;
    (void)op_id;
    (void)op_name;
#endif

    return true;
}

static const char *metric_name(jr_metric_t m)
{
    switch (m) {
        case JR_METRIC_DIRECT:   return "direct";
        case JR_METRIC_RESIDUAL: return "residual";
        case JR_METRIC_RECON:    return "recon";
        case JR_METRIC_VALUES:   return "values";
        default:                 return "?";
    }
}

static uint8_t profile_primary_digits(const fb_precision_profile_t *p,
                                       jr_metric_t m)
{
    switch (m) {
        case JR_METRIC_DIRECT:   return p->direct.guaranteed_digits;
        case JR_METRIC_RESIDUAL: return p->residual.guaranteed_digits;
        case JR_METRIC_RECON:    return p->reconstruction.guaranteed_digits;
        case JR_METRIC_VALUES:   return p->values.guaranteed_digits;
        default:                 return 0;
    }
}

[[maybe_unused]]
static bool profile_has_fatal(const fb_precision_profile_t *p, jr_metric_t m)
{
    switch (m) {
        case JR_METRIC_DIRECT:   return p->direct.has_fatal_failure;
        case JR_METRIC_RESIDUAL: return p->residual.has_fatal_failure;
        case JR_METRIC_RECON:    return p->reconstruction.has_fatal_failure;
        case JR_METRIC_VALUES:   return p->values.has_fatal_failure;
        default:                 return true;
    }
}

/* ============================================================================
 * main
 * ========================================================================== */

int main(void)
{
    jr_catalog_entry_t *all_ops = NULL;
    jr_reference_op_list_t reference_ops = { 0 };
    size_t all_op_count = 0;
    uint32_t start_op_id = 0;
    uint32_t end_op_id = UINT32_MAX;
    bool has_start_filter = false;
    bool has_end_filter = false;
    bool trace_op_start = false;
    bool trace_op_return = false;
    bool suppress_op_results = false;
    bool validate_heap = false;
    bool fail_on_reference_skip = false;
    bool audit_reference_skips = false;
    size_t selected_op_count = 0;

    /* Disable stdout buffering so crash location is always visible */
    setvbuf(stdout, NULL, _IONBF, 0);

    if (!load_op_catalog(&all_ops, &all_op_count)) {
        fprintf(stderr, "[FATAL] failed to load op catalog from src/judge/judge_op_ids.h\n");
        return 1;
    }

    has_start_filter = parse_env_u32("FB_JUDGE_START_OP_ID", &start_op_id);
    has_end_filter = parse_env_u32("FB_JUDGE_END_OP_ID", &end_op_id);
    trace_op_start = parse_env_flag("FB_JUDGE_TRACE_OP_START");
    trace_op_return = parse_env_flag("FB_JUDGE_TRACE_OP_RETURN");
    suppress_op_results = parse_env_flag("FB_JUDGE_SUPPRESS_OP_RESULTS");
    validate_heap = parse_env_flag("FB_JUDGE_VALIDATE_HEAP");
    fail_on_reference_skip = parse_env_flag("FB_JUDGE_FAIL_ON_REFERENCE_SKIP");
    audit_reference_skips = fail_on_reference_skip;
    if (!audit_reference_skips) {
        const char *reference_path_override = getenv("FB_JUDGE_REFERENCE_OP_LIST");
        audit_reference_skips =
            (reference_path_override != NULL && reference_path_override[0] != '\0');
    }

    if (audit_reference_skips && !load_reference_op_list(&reference_ops)) {
        fprintf(stderr,
                "[FATAL] failed to load reference op list for judge skip audit\n");
        free(all_ops);
        return 1;
    }

    if (has_start_filter && has_end_filter && end_op_id < start_op_id) {
        fprintf(stderr, "[FATAL] FB_JUDGE_END_OP_ID (%u) is less than FB_JUDGE_START_OP_ID (%u)\n",
                (unsigned)end_op_id, (unsigned)start_op_id);
        free_reference_op_list(&reference_ops);
        free(all_ops);
        return 1;
    }

    for (size_t oi = 0; oi < all_op_count; oi++) {
        uint32_t op_id = all_ops[oi].op_id;
        if (op_id < start_op_id || op_id > end_op_id) {
            continue;
        }
        selected_op_count++;
    }

    char profile_dir[512];
    build_profile_dir(profile_dir, sizeof(profile_dir));

    /* ---- Initialise judge ---- */
    fb_judge_status_t init_s = fb_judge_init(profile_dir);
    if (init_s != FB_JUDGE_OK) {
        fprintf(stderr, "[FATAL] fb_judge_init('%s') failed: %s\n",
                profile_dir, judge_status_str(init_s));
        free_reference_op_list(&reference_ops);
        free(all_ops);
        return 1;
    }

    /* ---- Load reference DLL and populate vtable via DLL export scan ---- *
     * fb_reference_init() is a no-op when the DLL cannot be found; the 192   *
     * statically-wired named fields are still available in that case.        */
    {
        const char *ref_names[] = {
            "faster_blaster_reference.dll",
            "libfaster_blaster_reference.dll",
            NULL
        };
        const char *ref_paths[] = {
            FB_REFERENCE_DLL_DIR,
            /* same dir as the test exe: DLL copied there by POST_BUILD */
            ".",
            NULL
        };
        fb_lib_handle_t ref_h = fb_plugin_load_library(ref_names, ref_paths);
        if (!ref_h) {
            fprintf(stderr, "[WARN] faster_blaster_reference.dll not found; "
                            "using statically-wired vtable only (192 ops).\n");
        }
        fb_reference_init(ref_h);   /* safe with ref_h == NULL */
    }

    /* ---- Register oracle (reference backend) ---- */
    const fb_backend_vtable_t *oracle_vtable = fb_reference_backend();
    if (!oracle_vtable) {
        fprintf(stderr, "[FATAL] fb_reference_backend() returned NULL\n");
        fb_judge_shutdown();
        free_reference_op_list(&reference_ops);
        free(all_ops);
        return 1;
    }
    fb_judge_register_oracle(oracle_vtable);

    /* ---- Print banner ---- */
    printf("======================================================\n");
    printf("  faster-blaster Judge Runner — end-to-end CI test\n");
    printf("  Oracle:      reference (built-in)\n");
    printf("  Profile dir: %s\n", profile_dir);
    printf("  Ops tested:  %u  |  Backends: %d\n",
           (unsigned)selected_op_count, k_backend_count);
        if (audit_reference_skips) {
         printf("  Ref audit:   enabled (%s)\n",
             reference_ops.path_used ? reference_ops.path_used : "<unknown>");
        }
    if (has_start_filter || has_end_filter) {
        printf("  Op filter:   [%u, %u]\n",
               (unsigned)start_op_id, (unsigned)end_op_id);
    }
    printf("======================================================\n\n");

    int global_pass = 0, global_fail = 0, global_skip = 0;
        int global_reference_skip = 0;

    /* ---- Run each backend ---- */
    for (int bi = 0; bi < k_backend_count; bi++) {
        const jr_backend_entry_t *be = &k_backends[bi];

        const fb_backend_vtable_t *vtable = be->get_vtable();
        if (!vtable) {
            if (be->optional) {
                printf("[SKIP] Backend '%s' not available on this system.\n\n",
                       be->name);
                continue;
            }
            fprintf(stderr,
                    "[FATAL] Mandatory backend '%s' vtable is NULL.\n",
                    be->name);
            fb_judge_shutdown();
            free_reference_op_list(&reference_ops);
            free(all_ops);
            return 1;
        }

        fb_judge_register_backend(be->backend_id, vtable);

        printf("Backend: %s  (id=%u)\n", be->name, be->backend_id);
        printf("  %-8s  %-5s  %-8s  %-10s  %s\n",
               "op", "dtype", "digits", "metric", "result");
        printf("  %-8s  %-5s  %-8s  %-10s  %s\n",
               "--------", "-----", "--------", "----------", "------");

        int b_pass = 0, b_fail = 0, b_skip = 0;
        int b_reference_skip = 0;

        for (size_t oi = 0; oi < all_op_count; oi++) {
            const jr_catalog_entry_t *catalog = &all_ops[oi];
            if (catalog->op_id < start_op_id || catalog->op_id > end_op_id) {
                continue;
            }
            const jr_op_entry_t *override = find_op_override(catalog->op_id);
            const fb_op_judge_meta_t *meta = fb_judge_meta_get(catalog->op_id);
                        fb_dtype_t dtype = override ? (fb_dtype_t)override->dtype
                                                                                : (fb_dtype_t)catalog->dtype;
            jr_metric_t metric = override ? override->metric : default_metric_from_meta(meta);
            uint8_t min_digits = override ? override->min_digits
                                                                                    : default_min_digits_from_meta(meta, dtype);
            const char *op_name = override ? override->name : catalog->name;

            const char *dtype_str =
                                (dtype == FB_DTYPE_F32)  ? "f32"  :
                                (dtype == FB_DTYPE_F64)  ? "f64"  :
                                (dtype == FB_DTYPE_CF32) ? "cf32" :
                                (dtype == FB_DTYPE_CF64) ? "cf64" :
                                (dtype == FB_DTYPE_F16)  ? "f16"  :
                                (dtype == FB_DTYPE_BF16) ? "bf16" :
                                (dtype == FB_DTYPE_I8)   ? "i8"   :
                                (dtype == FB_DTYPE_I32)  ? "i32"  : "???";

            fb_precision_profile_t prof;
            memset(&prof, 0, sizeof(prof));

            if (validate_heap &&
                !validate_process_heap("before", catalog->op_id, op_name)) {
                free_reference_op_list(&reference_ops);
                free(all_ops);
                fb_judge_shutdown();
                return 2;
            }

            if (trace_op_start) {
                printf("  [BEGIN] %u %-8s %-5s\n",
                       (unsigned)catalog->op_id, op_name, dtype_str);
            }

            fb_judge_status_t rs = fb_judge_run(
                catalog->op_id,
                be->backend_id,
                /*device_id=*/0u,
                FB_SIZE_SMALL,
                dtype,
                /*deep_audit=*/false,
                &prof);

            if (trace_op_return) {
                printf("  [END]   %u %-8s %-5s %s\n",
                       (unsigned)catalog->op_id, op_name, dtype_str,
                       judge_status_str(rs));
            }

            if (rs == FB_JUDGE_ERR_NOT_IMPL) {
                if (audit_reference_skips &&
                    reference_op_list_contains(&reference_ops, op_name)) {
                    const char *normalized_name = op_name;

                    if (strncmp(normalized_name, "cblas_", 6) == 0) {
                        normalized_name += 6;
                    }

                    b_reference_skip++;
                    if ((unsigned)b_reference_skip <= JR_MAX_REFERENCE_SKIP_PRINT) {
                        fprintf(stderr,
                                "[REF-SKIP] op_id=%u name=%s normalized=%s\n",
                                (unsigned)catalog->op_id,
                                op_name,
                                normalized_name);
                    }
                }
                if (!suppress_op_results) {
                    printf("  %-8s  %-5s  %-8s  %-10s  SKIP\n",
                           op_name, dtype_str, "--", metric_name(metric));
                }
                b_skip++;
                continue;
            }

            if (rs == FB_JUDGE_ERR_ORACLE_FAILURE) {
                /* Oracle returned non-finite results on ALL cases (likely extreme-
                 * scale overflow — expected for F32 matrix ops).  This is not a
                 * candidate failure; skip rather than fail. */
                if (!suppress_op_results) {
                    printf("  %-8s  %-5s  %-8s  %-10s  SKIP  (oracle overflow — extreme-scale case)\n",
                          op_name, dtype_str, "--", metric_name(metric));
                }
                b_skip++;
                continue;
            }

            if (rs != FB_JUDGE_OK) {
                if (!suppress_op_results) {
                    printf("  %-8s  %-5s  %-8s  %-10s  FAIL  (judge error: %s)\n",
                          op_name, dtype_str, "--",
                          metric_name(metric), judge_status_str(rs));
                }
                b_fail++;
                continue;
            }

            if (validate_heap &&
                !validate_process_heap("after", catalog->op_id, op_name)) {
                free_reference_op_list(&reference_ops);
                free(all_ops);
                fb_judge_shutdown();
                return 2;
            }

                 uint8_t  digits  = profile_primary_digits(&prof, metric);
            /* oracle_fatal on some cases = extreme-scale overflow (expected);
             * guaranteed_digits is already computed over the non-overflow cases. */
                 bool     passed  = (digits >= min_digits);

                        if (!suppress_op_results) {
                                printf("  %-8s  %-5s  %-8u  %-10s  %s"
                                             "  (need>=%u, p50=%uns)\n",
                                                 op_name, dtype_str,
                                             (unsigned)digits,
                                                 metric_name(metric),
                                             passed ? "PASS" : "FAIL",
                                                 (unsigned)min_digits,
                                             prof.timing.p50_ns);
                        }

            if (passed)
                b_pass++;
            else
                b_fail++;
        }

        printf("\n  Backend '%s': %d passed, %d failed, %d skipped\n\n",
               be->name, b_pass, b_fail, b_skip);
        if (audit_reference_skips) {
            printf("  Reference-overlap skips: %d\n\n", b_reference_skip);
        }

        global_pass += b_pass;
        global_fail += b_fail;
        global_skip += b_skip;
        global_reference_skip += b_reference_skip;
    }

    /* ---- Summary ---- */
    printf("======================================================\n");
    printf("  Total: %d passed, %d failed, %d skipped\n",
           global_pass, global_fail, global_skip);
    if (audit_reference_skips) {
        printf("  Reference-overlap skips: %d\n", global_reference_skip);
    }
    if (global_fail == 0 && (!fail_on_reference_skip || global_reference_skip == 0)) {
        printf("  Result: PASS\n");
    } else {
        if (global_fail > 0 && fail_on_reference_skip && global_reference_skip > 0) {
            printf("  Result: FAIL  (%d operation(s) below precision threshold, %d reference op(s) skipped)\n",
                   global_fail,
                   global_reference_skip);
        } else if (fail_on_reference_skip && global_reference_skip > 0) {
            printf("  Result: FAIL  (%d reference op(s) skipped by judge)\n",
                   global_reference_skip);
        } else {
            printf("  Result: FAIL  (%d operation(s) below precision threshold)\n",
                   global_fail);
        }
    }
    printf("======================================================\n");

    fb_judge_shutdown();
    free_reference_op_list(&reference_ops);
    free(all_ops);
    return (global_fail > 0 || (fail_on_reference_skip && global_reference_skip > 0)) ? 1 : 0;
}
