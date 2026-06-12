/**
 * @file conv_thunks.h
 * @brief Cross-convention calling-convention thunks for BLAS/LAPACK operations.
 *
 * Provides static per-op thunk arrays that bridge the two calling conventions
 * (@ref fb_conv_t) supported by faster-blaster:
 *
 *   - FB_CONV_CBLAS   — scalars passed by value, C linkage
 *   - FB_CONV_FORTRAN — all arguments passed by pointer, trailing underscore
 *
 * ### FB_CONV_REF — removed
 *
 * FB_CONV_REF was introduced under the assumption that the reference library
 * needed `saxpy_ref`-style names to avoid colliding with `cblas_saxpy` from
 * OpenBLAS or MKL in the same process.  That assumption is wrong.
 *
 * `fb_enumerate_and_populate()` resolves symbols via `dlsym(handle, …)` /
 * `GetProcAddress(handle, …)` using each backend's *own* library handle.
 * The same symbol name in two different loaded libraries produces two
 * distinct function pointers stored in two distinct vtables — no collision
 * at any level.  The vtable-per-backend architecture already provides
 * complete isolation; the `_ref` suffix adds nothing.
 *
 * The reference backend (faster-blaster-reference) should therefore export
 * standard `cblas_*` / `*_` names like any other backend, and
 * `FB_CONV_REF` is a dead classifier branch.  No CBLAS↔REF thunk is ever
 * needed — Strategy 5 aliases the slot with a plain pointer copy — but
 * the enum value itself is a candidate for removal in a future cleanup.
 *
 * ## Architecture
 *
 * ### Why globals at all?
 *
 * C has no closures: a statically compiled thunk function sits at a fixed
 * address and cannot capture a pointer to "its" vtable at compile time.
 * The per-op global arrays (`g_cblas_fn[]` / `g_fortran_fn[]`) are a
 * workaround — each thunk reads from the matching global slot to find its
 * target, and `fb_install_conv_thunks()` writes those slots from the vtable
 * being finalised.
 *
 * A cleaner alternative that avoids globals entirely is to perform the
 * convention conversion *inline at dispatch time*: if
 * `ext_ops[op][FB_CONV_FORTRAN]` is NULL but `ext_ops[op][FB_CONV_CBLAS]`
 * is not, the dispatch layer converts arguments and calls the CBLAS slot
 * directly — the vtable already holds its own target, no separate backing
 * store required.  That approach is the preferred upgrade path; thunks are
 * retained for the zero-overhead fast path when both slots are populated
 * from DLL exports.
 *
 * ### Single-backend constraint
 *
 * The current dispatch model finalises and routes through exactly one
 * selected vtable at a time (`fb_load_best_plugin()`).  Under that model
 * the globals are safe: `fb_install_conv_thunks()` is called once per
 * selected backend, and thunks always delegate to that backend's slot.
 *
 * Concurrently *active* backends sharing the same global would conflict.
 * The proper fix for true multi-backend concurrency is per-vtable backing:
 * add a `thunk_targets[FB_JUDGE_MAX_OPERATIONS]` field to the vtable and
 * point a single module-level `fb_backend_vtable_t *g_active_vtable` at
 * whichever backend is currently selected — thunks read through that
 * pointer rather than from flat global arrays.  Full per-vtable isolation
 * requires dynamically-generated trampolines (writable+executable memory
 * via `mprotect` / `VirtualAlloc`), which is left as a future option.
 *
 * ### Signature-shape policy
 *
 * Most thunk coverage should be expressed as reusable signature-shape
 * templates, not one macro per operation name. If two operations have the
 * same destination ABI and the same by-value/by-pointer parameter shape,
 * they should share one thunk template and differ only by type/op-id
 * instantiation.
 *
 * Current template inventory in `conv_thunks.c` is intentionally organized
 * around those fingerprints:
 *   - BLAS scalar/vector shape templates (AXPY, DOT, GEMV, GEMM, ...)
 *   - column-major LAPACK info-driver templates (for example LU factor,
 *     LU solve, and factored solve with transpose preservation)
 *   - symmetric/Hermitian LAPACK templates with UPLO remapping, plus
 *     dedicated row-major RHS translation when the family also carries a
 *     general right-hand-side matrix (for example POTRS and POSV)
 *   - positive-definite band templates with UPLO remapping, band-storage
 *     translation, and selective copy-back policy (for example PBCON,
 *     PBEQU, PBSV, PBSVX, PBTRF, PBTRS, PBRFS)
 *   - pivoted symmetric/Hermitian solve templates with workspace query,
 *     row-major matrix and RHS translation, and IPIV forwarding (for example
 *     SYSV and HESV)
 *   - matrix-plus-TAU factorization templates with workspace query and
 *     row-major matrix translation (for example GEQRF, GELQF, GEQLF, GERQF)
 *   - unblocked matrix-plus-TAU factorization templates with internal scratch
 *     but no workspace query (for example GELQ2, GEQL2, GERQ2)
 *   - general band matrix-vector templates with row-major band-storage
 *     translation and transpose forwarding (for example GBMVX)
 *   - general band equilibration templates with row-major band-storage
 *     translation and real scalar/vector outputs (for example GBEQU)
 *   - general band expert-solve templates with compact row-major AB,
 *     expanded factor storage, and hidden scratch propagation
 *     (for example GBSVX)
 *   - general band condition-estimate templates with expanded factor-storage
 *     translation (for example GBCON)
 *   - general band factorization templates with expanded factor-storage
 *     translation and row-major copy-back (for example GBTRF)
 *   - general band factored-solve templates with expanded factor-storage
 *     translation plus RHS matrix copy-back (for example GBTRS)
 *   - general band direct-solve templates with expanded factor-storage
 *     translation plus in-place AB and RHS copy-back (for example GBSV)
 *   - explicit Q-generator templates with matrix+k+TAU input, workspace query,
 *     and row-major matrix translation (for example ORGQR, UNGQR, ORGLQ)
 *   - Hessenberg reduction templates with matrix+ILO/IHI+TAU output,
 *     either workspace query or fixed N-sized scratch, and row-major matrix
 *     translation (for example GEHRD, GEHD2)
 *   - Hessenberg generator templates with matrix+ILO/IHI+TAU input,
 *     workspace query, and row-major matrix translation (for example ORGHR,
 *     UNGHR)
 *   - tridiagonal-reduction generator templates with UPLO remapping,
 *     workspace query, and row-major matrix translation (for example ORGTR,
 *     UNGTR)
 *   - unblocked generator templates with matrix+k+TAU input and internal
 *     scratch allocation but no workspace query (for example ORG2L, ORGL2,
 *     ORGR2, UNG2L)
 *   - reflector-application templates with side/trans, matrix+k+TAU input,
 *     workspace query, and row-major A/C translation (for example ORMQR,
 *     UNMQR, ORMLQ)
 *   - RZ reflector-application templates with side/trans plus extra `l`,
 *     either side-sized internal scratch or workspace query, and row-major
 *     A/C translation (for example ORMR3, UNMR3, ORMRZ, UNMRZ)
 *   - bidiagonal generator/application templates with vect selection,
 *     workspace query, and row-major matrix translation (for example ORGBR,
 *     UNGBR, ORMBR, UNMBR)
 *   - Hessenberg reflector-application templates with internal workspace
 *     query and column-major-only C bridges that reject unsupported row-major
 *     layout (for example ORMHR, UNMHR)
 *   - column-major LAPACK workspace-query templates (for example GEEV,
 *     GESVD, GESDD)
 *   - bidiagonal-reduction templates with D/E/TAUQ/TAUP outputs and
 *     workspace query in column-major layout (for example GEBRD)
 *   - layout-aware symmetric/Hermitian templates with UPLO remapping
 *   - symmetric/Hermitian band equilibration, condition-estimation,
 *     refinement, and direct-solve templates with band-storage transpose,
 *     selective row-major copy-back, real scalar outputs, and fixed internal
 *     scratch where required (for example PBEQU, PBCON, PBRFS, PBSV)
 *
 * The fingerprint for choosing a thunk template therefore includes more than
 * the raw C types. It must also include policy dimensions such as layout
 * handling, workspace-query behavior, INFO propagation, and whether mode flags
 * like `jobz`/`jobu`/`jobvt` are forwarded, remapped, or unsupported.
 *
 * A bespoke thunk is warranted only when the family needs additional policy,
 * for example:
 *   - row-major to column-major layout translation
 *   - UPLO/SIDE/TRANS remapping beyond plain pointer wrapping
 *   - workspace-query handling (`lwork=-1`, real-part extraction, etc.)
 *   - extra output arrays or INFO propagation rules that do not match an
 *     existing template
 *
 * ### Layout policy for generated LAPACK C bridges
 *
 * `FB_CONV_FORTRAN` exports are always column-major. A generated
 * `FB_CONV_CBLAS` bridge from an underscore-only LAPACK export must not claim
 * row-major support unless the thunk explicitly translates layout and that
 * translation is covered by tests. When no translation exists, the generated C
 * bridge must behave as column-major only and reject row-major calls instead of
 * silently forwarding incorrect storage to the Fortran kernel.
 *
 * ## Usage (in vtable_autofill.c — Strategy 5)
 *
 * ```c
 * #include "conv_thunks.h"
 *
 * // After Strategies 1-4, fill empty convention slots with thunks:
 * for (uint32_t op = 0; op < FB_JUDGE_MAX_OPERATIONS; op++) {
 *     fb_install_conv_thunks(vtable, op);
 * }
 * ```
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#pragma once

#include <stdint.h>
#include "../backends/backend_interface.h"   /* fb_backend_vtable_t, fb_conv_t */
#include "../judge/judge_op_ids.h"           /* FB_OP_* constants             */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Pre-built thunk: CBLAS → Fortran convention.
 *
 * `k_cblas_to_fortran_thunks[op]` is a fb_generic_fn that wraps the CBLAS
 * implementation of @p op and exposes it with Fortran linkage (all-pointer).
 * NULL when no thunk is available for the given op.
 */
extern fb_generic_fn const k_cblas_to_fortran_thunks[FB_JUDGE_MAX_OPERATIONS];

/**
 * @brief Pre-built thunk: Fortran → CBLAS convention.
 *
 * `k_fortran_to_cblas_thunks[op]` is a fb_generic_fn that wraps the Fortran
 * implementation of @p op and exposes it with CBLAS linkage (scalars by value).
 * NULL when no thunk is available for the given op.
 */
extern fb_generic_fn const k_fortran_to_cblas_thunks[FB_JUDGE_MAX_OPERATIONS];

/**
 * @brief Populate per-op global backing pointers and patch any empty convention
 *        slots in @p vtable for the given @p op_id using the appropriate thunk.
 *
 * Called by `fb_finalize_plugin_vtable()` (Strategy 5) for every op after
 * Strategies 1-4 have run.  Safe to call when both conv slots are already
 * filled — no-op in that case.
 *
 * @param vtable   The vtable being finalised.  Must not be NULL.
 * @param op_id    Operation index (0 … FB_JUDGE_MAX_OPERATIONS-1).
 */
void fb_install_conv_thunks(fb_backend_vtable_t *vtable, uint32_t op_id);

#ifdef __cplusplus
}
#endif
