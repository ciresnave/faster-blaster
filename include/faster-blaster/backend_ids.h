/**
 * @file backend_ids.h
 * @brief Stable numeric backend identifiers.
 *
 * These IDs are embedded in .fbjp judge profile files on disk.  They MUST
 * remain constant across releases so that stored profiles remain valid
 * between software updates.
 *
 * Assignment rule: IDs are assigned once, sequentially, and documented here.
 * Do NOT renumber existing entries.  New backends receive the next available
 * integer.  Remove a backend by marking it RESERVED — never reuse an ID.
 *
 * The IDs are used in three places:
 *   1. When the bench/judge CLI stores a profile:
 *        fb_judge_run(dir, FB_BACKEND_ID_OPENBLAS, device, op, ...)
 *   2. When fb_init() builds the dispatch table from stored profiles:
 *        fb_judge_build_dispatch_table(..., backend_ids, n, FB_BACKEND_ID_REFERENCE, ...)
 *   3. When the operation router looks up the preferred backend for an op:
 *        uint32_t id = fb_judge_get_routed_backend_id(FB_OP_SAXPY);
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_BACKEND_IDS_H
#define FASTER_BLASTER_BACKEND_IDS_H

#include <stdint.h>

/* =========================================================================
 * CPU backends
 * ========================================================================= */

/** AMD AOCL-BLIS — vendor-optimised BLIS for Zen microarchitectures. */
#define FB_BACKEND_ID_AOCL_BLIS     0u

/** Upstream BLIS — architecture-generic upstream BLIS library. */
#define FB_BACKEND_ID_BLIS          1u

/** OpenBLAS — open-source optimised BLAS/LAPACK. */
#define FB_BACKEND_ID_OPENBLAS      2u

/** Intel oneMKL — Intel Math Kernel Library. */
#define FB_BACKEND_ID_MKL           3u

/** Apple Accelerate / vecLib — macOS / iOS framework. */
#define FB_BACKEND_ID_ACCELERATE    4u

/* =========================================================================
 * GPU backends
 * ========================================================================= */

/** NVIDIA cuBLAS — CUDA-based BLAS for NVIDIA GPUs. */
#define FB_BACKEND_ID_CUBLAS        5u

/** AMD rocBLAS — ROCm-based BLAS for AMD GPUs. */
#define FB_BACKEND_ID_ROCBLAS       6u

/** Intel oneMKL SYCL — SYCL-based linear algebra for Intel GPUs. */
#define FB_BACKEND_ID_ONEMKL        7u

/** Apple Metal / MPSMatrix — Metal Performance Shaders on Apple Silicon. */
#define FB_BACKEND_ID_METAL         8u

/** CLBlast — OpenCL BLAS for any OpenCL-capable GPU. */
#define FB_BACKEND_ID_CLBLAST       9u

/** clBLAS — legacy AMD clBLAS for OpenCL devices. */
#define FB_BACKEND_ID_CLBLAS        10u

/* =========================================================================
 * Reference / fallback
 * ========================================================================= */

/**
 * faster-blaster-reference — portable pure-C reference implementation.
 * Always available; used as fallback and correctness oracle.
 */
#define FB_BACKEND_ID_REFERENCE     11u

/**
 * faster-blaster-reference DLL — dynamic loader for the external
 * faster_blaster_reference.dll / .so.  Score 10; higher than the static
 * reference (score 5) but lower than any optimised backend.
 * Available only when the DLL is found on the system.
 */
#define FB_BACKEND_ID_BLR           12u

/* =========================================================================
 * Sentinel
 * ========================================================================= */

/**
 * No backend selected.  Returned by fb_judge_get_routed_backend_id() when
 * the judge dispatch table has not been loaded or no profile exists for the
 * requested operation.
 */
#define FB_BACKEND_ID_NONE          UINT32_MAX

/* =========================================================================
 * Next available ID (update when adding a new backend above)
 * ========================================================================= */
#define FB_BACKEND_ID__NEXT_FREE    14u

#endif /* FASTER_BLASTER_BACKEND_IDS_H */
