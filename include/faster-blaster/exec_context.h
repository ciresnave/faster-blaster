/**
 * @file exec_context.h
 * @brief Unified execution context for mixed local/distributed operation sequences.
 *
 * When the sequence planner (op_chain / judge) schedules a sequence of operations
 * it needs to know which execution environments are available.  This context is
 * threaded through the entire dispatch path so that:
 *
 *  - BLAS/LAPACK ops route to the vtable of whichever backend is selected
 *  - GPU ops route through the gpu context's stream/handle
 *  - ScaLAPACK ops route through the distributed context's BLACS handle
 *
 * The judge reasons over ALL op IDs (BLAS L1/L2/L3, LAPACK, ScaLAPACK) in
 * one flat sequence.  The execution layer consults this context to branch:
 *
 *   if (op_id >= FB_OP_SCALAPACK_BASE)   → fb_scalapack_dispatch_op()
 *   else if (ctx->gpu && is_gpu_op)       → vtable gpu path
 *   else                                  → vtable cpu path
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_EXEC_CONTEXT_H
#define FASTER_BLASTER_EXEC_CONTEXT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations — concrete definitions in gpu_backend_trait.h and
 * scalapack.h respectively.  exec_context.h itself pulls in neither, keeping
 * the dependency graph clean. */
typedef struct fb_gpu_context  fb_gpu_context_t;
typedef struct fb_distributed_ctx fb_distributed_ctx_t;

/**
 * @brief Unified execution context for a scheduled operation sequence.
 *
 * Allocate on the stack or heap; initialize each non-NULL member before
 * passing to the sequence executor.  Members set to NULL indicate that the
 * corresponding execution environment is not in play for this sequence.
 *
 * Example — CPU-only sequence:
 * @code
 *   fb_exec_context_t ctx = { .gpu = NULL, .dist = NULL };
 * @endcode
 *
 * Example — GPU-accelerated local sequence:
 * @code
 *   fb_gpu_context_t gpu_ctx;
 *   fb_gpu_init(&gpu_ctx, 0);            // device 0
 *   fb_exec_context_t ctx = { .gpu = &gpu_ctx, .dist = NULL };
 * @endcode
 *
 * Example — Mixed local-BLAS + ScaLAPACK distributed sequence:
 * @code
 *   fb_distributed_ctx_t dist;
 *   fb_distributed_ctx_init(&dist, MPI_COMM_WORLD, 4, 4);
 *   fb_exec_context_t ctx = { .gpu = NULL, .dist = &dist };
 * @endcode
 */
typedef struct fb_exec_context {
    /** GPU execution context.  NULL when no GPU is used in this sequence. */
    fb_gpu_context_t     *gpu;

    /** Distributed (BLACS/MPI) execution context.  NULL for local sequences. */
    fb_distributed_ctx_t *dist;
} fb_exec_context_t;

/** Convenience: a context with no GPU and no distributed environment. */
#define FB_EXEC_CONTEXT_LOCAL  ((fb_exec_context_t){ .gpu = NULL, .dist = NULL })

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_EXEC_CONTEXT_H */
