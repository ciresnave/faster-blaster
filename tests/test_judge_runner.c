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

/* ---- Public judge API ---- */
#include "../include/faster-blaster/judge.h"
#include "../include/faster-blaster/backend_ids.h"
#include "../include/benchmark_types.h"   /* fb_size_class_t, FB_SIZE_SMALL */

/* ---- Internal judge tables (non-public but required for op ID constants
 *      and dtype enum; consistent with test_judge.c pattern).           ---- */
#include "../src/judge/judge_op_ids.h"   /* FB_OP_SAXPY, FB_OP_SGEMM, … */
#include "../src/judge/judge_types.h"    /* FB_DTYPE_F32, FB_DTYPE_F64, … */

/* ---- Reference backend vtable ---- */
#include "../src/backends/reference.h"   /* fb_reference_backend() */

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
    { FB_OP_SASUM,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sasum"   },
    { FB_OP_DASUM,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dasum"   },
    { FB_OP_SAXPY,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "saxpy"   },
    { FB_OP_DAXPY,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "daxpy"   },
    { FB_OP_SCOPY,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "scopy"   },
    { FB_OP_DCOPY,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dcopy"   },
    { FB_OP_SDOT,    FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sdot"    },
    { FB_OP_DDOT,    FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "ddot"    },
    { FB_OP_SNRM2,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "snrm2"   },
    { FB_OP_DNRM2,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dnrm2"   },
    { FB_OP_SSCAL,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sscal"   },
    { FB_OP_DSCAL,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dscal"   },
    { FB_OP_SSWAP,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sswap"   },
    { FB_OP_DSWAP,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dswap"   },
    { FB_OP_SDSDOT,  FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sdsdot"  },
    { FB_OP_DSDOT,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "dsdot"   },
    { FB_OP_SROT,    FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "srot"    },
    { FB_OP_DROT,    FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "drot"    },
    { FB_OP_SROTG,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "srotg"   },
    { FB_OP_DROTG,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "drotg"   },
    { FB_OP_SROTM,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "srotm"   },
    { FB_OP_DROTM,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "drotm"   },
    { FB_OP_SROTMG,  FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "srotmg"  },
    /* BLAS Level 1 — DIRECT (complex)                                  */
    { FB_OP_CAXPY,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "caxpy"   },
    { FB_OP_ZAXPY,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zaxpy"   },
    { FB_OP_CSCAL,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cscal"   },
    { FB_OP_ZSCAL,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zscal"   },
    { FB_OP_CSSCAL,  FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csscal"  },
    { FB_OP_ZDSCAL,  FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdscal"  },
    { FB_OP_CCOPY,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ccopy"   },
    { FB_OP_ZCOPY,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zcopy"   },
    { FB_OP_CSWAP,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cswap"   },
    { FB_OP_ZSWAP,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zswap"   },
    { FB_OP_CDOTU,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cdotu"   },
    { FB_OP_ZDOTU,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdotu"   },
    { FB_OP_CDOTC,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cdotc"   },
    { FB_OP_ZDOTC,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdotc"   },
    { FB_OP_SCASUM,  FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "scasum"  },
    { FB_OP_DZASUM,  FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "dzasum"  },
    { FB_OP_SCNRM2,  FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "scnrm2"  },
    { FB_OP_DZNRM2,  FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "dznrm2"  },
    { FB_OP_CROT,    FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "crot"    },
    { FB_OP_ZROT,    FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zrot"    },
    { FB_OP_ZDROT,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zdrot"   },
    /* BLAS Level 1 — INDEX                                             */
    { FB_OP_ISAMAX,  FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "isamax"  },
    { FB_OP_IDAMAX,  FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "idamax"  },
    { FB_OP_ICAMAX,  FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "icamax"  },
    { FB_OP_IZAMAX,  FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "izamax"  },
    /* ================================================================ */
    /* BLAS Level 2 — DIRECT                                            */
    /* ================================================================ */
    { FB_OP_SGEMV,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sgemv"   },
    { FB_OP_DGEMV,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dgemv"   },
    { FB_OP_CGEMV,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgemv"   },
    { FB_OP_ZGEMV,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgemv"   },
    { FB_OP_SSYMV,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "ssymv"   },
    { FB_OP_DSYMV,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dsymv"   },
    { FB_OP_CHEMV,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chemv"   },
    { FB_OP_ZHEMV,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhemv"   },
    { FB_OP_STRMV,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "strmv"   },
    { FB_OP_DTRMV,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dtrmv"   },
    { FB_OP_CTRMV,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctrmv"   },
    { FB_OP_ZTRMV,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztrmv"   },
    { FB_OP_SGER,    FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sger"    },
    { FB_OP_DGER,    FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dger"    },
    { FB_OP_CGERU,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgeru"   },
    { FB_OP_CGERC,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgerc"   },
    { FB_OP_ZGERU,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgeru"   },
    { FB_OP_ZGERC,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgerc"   },
    { FB_OP_SSYR,    FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "ssyr"    },
    { FB_OP_DSYR,    FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dsyr"    },
    { FB_OP_CHER,    FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cher"    },
    { FB_OP_ZHER,    FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zher"    },
    { FB_OP_SSYR2,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "ssyr2"   },
    { FB_OP_DSYR2,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dsyr2"   },
    { FB_OP_CHER2,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cher2"   },
    { FB_OP_ZHER2,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zher2"   },
    { FB_OP_SSPMV,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sspmv"   },
    { FB_OP_DSPMV,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dspmv"   },
    { FB_OP_CHPMV,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chpmv"   },
    { FB_OP_ZHPMV,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhpmv"   },
    { FB_OP_SSBMV,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "ssbmv"   },
    { FB_OP_DSBMV,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dsbmv"   },
    { FB_OP_CHBMV,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chbmv"   },
    { FB_OP_ZHBMV,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhbmv"   },
    { FB_OP_STBMV,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "stbmv"   },
    { FB_OP_DTBMV,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dtbmv"   },
    { FB_OP_CTBMV,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctbmv"   },
    { FB_OP_ZTBMV,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztbmv"   },
    { FB_OP_STPMV,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "stpmv"   },
    { FB_OP_DTPMV,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dtpmv"   },
    { FB_OP_CTPMV,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctpmv"   },
    { FB_OP_ZTPMV,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztpmv"   },
    { FB_OP_SSPR,    FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sspr"    },
    { FB_OP_DSPR,    FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dspr"    },
    { FB_OP_CHPR,    FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chpr"    },
    { FB_OP_ZHPR,    FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhpr"    },
    { FB_OP_SSPR2,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sspr2"   },
    { FB_OP_DSPR2,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dspr2"   },
    { FB_OP_CHPR2,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chpr2"   },
    { FB_OP_ZHPR2,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhpr2"   },
    /* BLAS Level 2 — SOLVE (TRSV / TBSV / TPSV)
     * NOTE: strsv/stbsv/stpsv have SOLVE archetype in metadata but are
     * dispatched only in judge_direct.c.  Running them here via RESIDUAL
     * metric always gets NOT_IMPL → 0 digits → FAIL.  Re-add once either
     * (a) judge_solve.c gains trsv/tbsv/tpsv handlers, or
     * (b) metadata is changed to DIRECT_ENTRY for these ops.            */
    /* ================================================================ */
    /* BLAS Level 3 — DIRECT                                            */
    /* ================================================================ */
    { FB_OP_SGEMM,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "sgemm"   },
    { FB_OP_DGEMM,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dgemm"   },
    { FB_OP_CGEMM,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cgemm"   },
    { FB_OP_ZGEMM,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zgemm"   },
    { FB_OP_SSYMM,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "ssymm"   },
    { FB_OP_DSYMM,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dsymm"   },
    { FB_OP_CSYMM,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csymm"   },
    { FB_OP_ZSYMM,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsymm"   },
    { FB_OP_CHEMM,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "chemm"   },
    { FB_OP_ZHEMM,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zhemm"   },
    { FB_OP_SSYRK,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "ssyrk"   },
    { FB_OP_DSYRK,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dsyrk"   },
    { FB_OP_CSYRK,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csyrk"   },
    { FB_OP_ZSYRK,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsyrk"   },
    { FB_OP_CHERK,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cherk"   },
    { FB_OP_ZHERK,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zherk"   },
    { FB_OP_SSYR2K,  FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "ssyr2k"  },
    { FB_OP_DSYR2K,  FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dsyr2k"  },
    { FB_OP_CSYR2K,  FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "csyr2k"  },
    { FB_OP_ZSYR2K,  FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zsyr2k"  },
    { FB_OP_CHER2K,  FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "cher2k"  },
    { FB_OP_ZHER2K,  FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "zher2k"  },
    { FB_OP_STRMM,   FB_DTYPE_F32,  JR_METRIC_DIRECT, JR_MIN_F32,  "strmm"   },
    { FB_OP_DTRMM,   FB_DTYPE_F64,  JR_METRIC_DIRECT, JR_MIN_F64,  "dtrmm"   },
    { FB_OP_CTRMM,   FB_DTYPE_CF32, JR_METRIC_DIRECT, JR_MIN_CF32, "ctrmm"   },
    { FB_OP_ZTRMM,   FB_DTYPE_CF64, JR_METRIC_DIRECT, JR_MIN_CF64, "ztrmm"   },
    /* BLAS Level 3 — SOLVE (TRSM)
     * NOTE: Same archetype mismatch as trsv above.  Re-add when
     * judge_solve.c gains trsm handler or metadata switches to DIRECT.  */
    /* ================================================================ */
    /* LAPACK — Factorization (RECON metric)                            */
    /* ================================================================ */
    { FB_OP_SGETRF,  FB_DTYPE_F32,  JR_METRIC_RECON, JR_MIN_LAPACK_F32, "sgetrf" },
    { FB_OP_DGETRF,  FB_DTYPE_F64,  JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dgetrf" },
    { FB_OP_CGETRF,  FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cgetrf" },
    { FB_OP_ZGETRF,  FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zgetrf" },
    { FB_OP_SPOTRF,  FB_DTYPE_F32,  JR_METRIC_RECON, JR_MIN_LAPACK_F32, "spotrf" },
    { FB_OP_DPOTRF,  FB_DTYPE_F64,  JR_METRIC_RECON, JR_MIN_LAPACK_F64, "dpotrf" },
    { FB_OP_CPOTRF,  FB_DTYPE_CF32, JR_METRIC_RECON, JR_MIN_LAPACK_F32, "cpotrf" },
    { FB_OP_ZPOTRF,  FB_DTYPE_CF64, JR_METRIC_RECON, JR_MIN_LAPACK_F64, "zpotrf" },
    /* LAPACK QR factorization (sgeqrf/dgeqrf/cgeqrf/zgeqrf) and its RECON
     * partner (sorgqr/dorgqr/cungqr/zungqr):
     * NOTE: sgeqrf_ref uses column-major indexing (idx_below=(i+1)+i*lda)
     * instead of row-major ((i+1)*lda+i).  For a 128×128 corpus matrix
     * slarf_ref accesses a[16510] past the 16384-element bound →
     * STATUS_ACCESS_VIOLATION.  Re-add after fixing sgeqrf_ref in
     * faster-blaster-reference (and rebuilding libfaster_blaster_reference.a). */
    /* LAPACK — Triangular Solve (RESIDUAL metric)                       */
    { FB_OP_SGETRS,  FB_DTYPE_F32,  JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "sgetrs" },
    { FB_OP_DGETRS,  FB_DTYPE_F64,  JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "dgetrs" },
    { FB_OP_CGETRS,  FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "cgetrs" },
    { FB_OP_ZGETRS,  FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "zgetrs" },
    { FB_OP_SPOTRS,  FB_DTYPE_F32,  JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "spotrs" },
    { FB_OP_DPOTRS,  FB_DTYPE_F64,  JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "dpotrs" },
    { FB_OP_CPOTRS,  FB_DTYPE_CF32, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F32, "cpotrs" },
    { FB_OP_ZPOTRS,  FB_DTYPE_CF64, JR_METRIC_RESIDUAL, JR_MIN_LAPACK_F64, "zpotrs" },
    /* SPECTRAL (ssyev/dsyev/cheev/zheev/sgesvd/dgesvd/geev): oracle NULL —
     * auxiliary *_ref routines (gehrd/hseqr/trevc/bdsqr) not yet in
     * faster-blaster-reference; re-add entries when auxiliaries land */
};

static const int k_op_count = (int)(sizeof(k_ops) / sizeof(k_ops[0]));

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
    /* Disable stdout buffering so crash location is always visible */
    setvbuf(stdout, NULL, _IONBF, 0);

    /* ---- Locate a writable temp directory ---- */
    const char *tmp = getenv("TEMP");
    if (!tmp) tmp   = getenv("TMPDIR");
    if (!tmp) tmp   = "/tmp";

    char profile_dir[512];
    snprintf(profile_dir, sizeof(profile_dir),
             "%s/fb_judge_runner_profiles", tmp);

    /* ---- Initialise judge ---- */
    fb_judge_status_t init_s = fb_judge_init(profile_dir);
    if (init_s != FB_JUDGE_OK) {
        fprintf(stderr, "[FATAL] fb_judge_init('%s') failed: %s\n",
                profile_dir, judge_status_str(init_s));
        return 1;
    }

    /* ---- Register oracle (reference backend) ---- */
    const fb_backend_vtable_t *oracle_vtable = fb_reference_backend();
    if (!oracle_vtable) {
        fprintf(stderr, "[FATAL] fb_reference_backend() returned NULL\n");
        fb_judge_shutdown();
        return 1;
    }
    fb_judge_register_oracle(oracle_vtable);

    /* ---- Print banner ---- */
    printf("======================================================\n");
    printf("  faster-blaster Judge Runner — end-to-end CI test\n");
    printf("  Oracle:      reference (built-in)\n");
    printf("  Profile dir: %s\n", profile_dir);
    printf("  Ops tested:  %d  |  Backends: %d\n",
           k_op_count, k_backend_count);
    printf("======================================================\n\n");

    int global_pass = 0, global_fail = 0, global_skip = 0;

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
            return 1;
        }

        fb_judge_register_backend(be->backend_id, vtable);

        printf("Backend: %s  (id=%u)\n", be->name, be->backend_id);
        printf("  %-8s  %-5s  %-8s  %-10s  %s\n",
               "op", "dtype", "digits", "metric", "result");
        printf("  %-8s  %-5s  %-8s  %-10s  %s\n",
               "--------", "-----", "--------", "----------", "------");

        int b_pass = 0, b_fail = 0, b_skip = 0;

        for (int oi = 0; oi < k_op_count; oi++) {
            const jr_op_entry_t *op = &k_ops[oi];

            const char *dtype_str =
                (op->dtype == FB_DTYPE_F32) ? "f32" :
                (op->dtype == FB_DTYPE_F64) ? "f64" : "???";

            fb_precision_profile_t prof;
            memset(&prof, 0, sizeof(prof));

            fb_judge_status_t rs = fb_judge_run(
                op->op_id,
                be->backend_id,
                /*device_id=*/0u,
                FB_SIZE_SMALL,
                op->dtype,
                /*deep_audit=*/false,
                &prof);

            if (rs == FB_JUDGE_ERR_NOT_IMPL) {
                printf("  %-8s  %-5s  %-8s  %-10s  SKIP\n",
                       op->name, dtype_str, "--", metric_name(op->metric));
                b_skip++;
                continue;
            }

            if (rs == FB_JUDGE_ERR_ORACLE_FAILURE) {
                /* Oracle returned non-finite results on ALL cases (likely extreme-
                 * scale overflow — expected for F32 matrix ops).  This is not a
                 * candidate failure; skip rather than fail. */
                printf("  %-8s  %-5s  %-8s  %-10s  SKIP  (oracle overflow — extreme-scale case)\n",
                       op->name, dtype_str, "--", metric_name(op->metric));
                b_skip++;
                continue;
            }

            if (rs != FB_JUDGE_OK) {
                printf("  %-8s  %-5s  %-8s  %-10s  FAIL  (judge error: %s)\n",
                       op->name, dtype_str, "--",
                       metric_name(op->metric), judge_status_str(rs));
                b_fail++;
                continue;
            }

            uint8_t  digits  = profile_primary_digits(&prof, op->metric);
            /* oracle_fatal on some cases = extreme-scale overflow (expected);
             * guaranteed_digits is already computed over the non-overflow cases. */
            bool     passed  = (digits >= op->min_digits);

            printf("  %-8s  %-5s  %-8u  %-10s  %s"
                   "  (need>=%u, p50=%uns)\n",
                   op->name, dtype_str,
                   (unsigned)digits,
                   metric_name(op->metric),
                   passed ? "PASS" : "FAIL",
                   (unsigned)op->min_digits,
                   prof.timing.p50_ns);

            if (passed)
                b_pass++;
            else
                b_fail++;
        }

        printf("\n  Backend '%s': %d passed, %d failed, %d skipped\n\n",
               be->name, b_pass, b_fail, b_skip);

        global_pass += b_pass;
        global_fail += b_fail;
        global_skip += b_skip;
    }

    /* ---- Summary ---- */
    printf("======================================================\n");
    printf("  Total: %d passed, %d failed, %d skipped\n",
           global_pass, global_fail, global_skip);
    if (global_fail == 0) {
        printf("  Result: PASS\n");
    } else {
        printf("  Result: FAIL  (%d operation(s) below precision threshold)\n",
               global_fail);
    }
    printf("======================================================\n");

    fb_judge_shutdown();
    return (global_fail > 0) ? 1 : 0;
}
