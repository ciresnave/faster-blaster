/**
 * @file dispatch.h
 * @brief Zero-overhead function dispatch system
 * 
 * This module implements the core dispatch mechanism that routes BLAS function
 * calls to the optimal backend. After calibration, dispatch overhead is
 * effectively zero - just a single indirect function call.
 * 
 * The dispatch system supports:
 * - Direct function pointer dispatch (zero overhead after init)
 * - Runtime backend reconfiguration
 * - Multi-dimensional dispatch keys (operation, size, dtype, layout)
 * - Thread-safe concurrent access
 * - Statistics collection (optional, minimal overhead)
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_DISPATCH_H
#define FB_DISPATCH_H

#include "../backends/backend_interface.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct fb_calibration_database_t;

/* ============================================================================
 * Dispatch Table Types
 * ========================================================================= */

/**
 * @brief Dispatch policy - how backends are selected
 */
typedef enum {
    FB_DISPATCH_MANUAL = 0,      /**< User explicitly selects backend */
    FB_DISPATCH_AUTO,            /**< Auto-select based on hardware detection */
    FB_DISPATCH_CALIBRATED       /**< Use calibration data to select optimal backend */
} fb_dispatch_policy_t;

/* Maximum number of registered backends */
#define MAX_BACKENDS 16

/**
 * @brief Dispatch table structure
 * 
 * This is now a complete definition (not opaque) so dispatch.c can access members.
 */
typedef struct fb_dispatch_table_t {
    const fb_backend_vtable_t *backends[MAX_BACKENDS];
    const char *backend_names[MAX_BACKENDS];
    size_t num_backends;
    size_t active_backend_idx;
    fb_dispatch_policy_t policy;
    bool initialized;
} fb_dispatch_table_t;

/**
 * @brief Operation identifier for dispatch (212 operations)
 */
typedef enum {
    /* ===== BLAS Level 1 (52 operations) ===== */
    FB_OP_SROTG = 0, FB_OP_DROTG, FB_OP_CROTG, FB_OP_ZROTG,
    FB_OP_SROTMG, FB_OP_DROTMG,
    FB_OP_SROT, FB_OP_DROT, FB_OP_CROT, FB_OP_ZROT,
    FB_OP_SROTM, FB_OP_DROTM,
    FB_OP_SSWAP, FB_OP_DSWAP, FB_OP_CSWAP, FB_OP_ZSWAP,
    FB_OP_SSCAL, FB_OP_DSCAL, FB_OP_CSCAL, FB_OP_ZSCAL, FB_OP_CSSCAL, FB_OP_ZDSCAL,
    FB_OP_SCOPY, FB_OP_DCOPY, FB_OP_CCOPY, FB_OP_ZCOPY,
    FB_OP_SAXPY, FB_OP_DAXPY, FB_OP_CAXPY, FB_OP_ZAXPY,
    FB_OP_SDOT, FB_OP_DDOT, FB_OP_SDSDOT, FB_OP_DSDOT,
    FB_OP_CDOTU, FB_OP_ZDOTU,
    FB_OP_CDOTC, FB_OP_ZDOTC,
    FB_OP_SNRM2, FB_OP_DNRM2, FB_OP_SCNRM2, FB_OP_DZNRM2,
    FB_OP_SASUM, FB_OP_DASUM, FB_OP_SCASUM, FB_OP_DZASUM,
    FB_OP_ISAMAX, FB_OP_IDAMAX, FB_OP_ICAMAX, FB_OP_IZAMAX,
    
    /* ===== BLAS Level 2 (70 operations) ===== */
    FB_OP_SGEMV, FB_OP_DGEMV, FB_OP_CGEMV, FB_OP_ZGEMV,
    FB_OP_SGBMV, FB_OP_DGBMV, FB_OP_CGBMV, FB_OP_ZGBMV,
    FB_OP_CHEMV, FB_OP_ZHEMV,
    FB_OP_CHBMV, FB_OP_ZHBMV,
    FB_OP_CHPMV, FB_OP_ZHPMV,
    FB_OP_SSYMV, FB_OP_DSYMV,
    FB_OP_SSBMV, FB_OP_DSBMV,
    FB_OP_SSPMV, FB_OP_DSPMV,
    FB_OP_STRMV, FB_OP_DTRMV, FB_OP_CTRMV, FB_OP_ZTRMV,
    FB_OP_STBMV, FB_OP_DTBMV, FB_OP_CTBMV, FB_OP_ZTBMV,
    FB_OP_STPMV, FB_OP_DTPMV, FB_OP_CTPMV, FB_OP_ZTPMV,
    FB_OP_STRSV, FB_OP_DTRSV, FB_OP_CTRSV, FB_OP_ZTRSV,
    FB_OP_STBSV, FB_OP_DTBSV, FB_OP_CTBSV, FB_OP_ZTBSV,
    FB_OP_STPSV, FB_OP_DTPSV, FB_OP_CTPSV, FB_OP_ZTPSV,
    FB_OP_SGER, FB_OP_DGER,
    FB_OP_CGERU, FB_OP_ZGERU,
    FB_OP_CGERC, FB_OP_ZGERC,
    FB_OP_CHER, FB_OP_ZHER,
    FB_OP_CHPR, FB_OP_ZHPR,
    FB_OP_CHER2, FB_OP_ZHER2,
    FB_OP_CHPR2, FB_OP_ZHPR2,
    FB_OP_SSYR, FB_OP_DSYR,
    FB_OP_SSPR, FB_OP_DSPR,
    FB_OP_SSYR2, FB_OP_DSYR2,
    FB_OP_SSPR2, FB_OP_DSPR2,
    
    /* ===== BLAS Level 3 (30 operations) ===== */
    FB_OP_SGEMM, FB_OP_DGEMM, FB_OP_CGEMM, FB_OP_ZGEMM,
    FB_OP_SSYMM, FB_OP_DSYMM, FB_OP_CSYMM, FB_OP_ZSYMM,
    FB_OP_CHEMM, FB_OP_ZHEMM,
    FB_OP_SSYRK, FB_OP_DSYRK, FB_OP_CSYRK, FB_OP_ZSYRK,
    FB_OP_CHERK, FB_OP_ZHERK,
    FB_OP_SSYR2K, FB_OP_DSYR2K, FB_OP_CSYR2K, FB_OP_ZSYR2K,
    FB_OP_CHER2K, FB_OP_ZHER2K,
    FB_OP_STRMM, FB_OP_DTRMM, FB_OP_CTRMM, FB_OP_ZTRMM,
    FB_OP_STRSM, FB_OP_DTRSM, FB_OP_CTRSM, FB_OP_ZTRSM,
    
    /* ===== LAPACK Subset (60 operations) ===== */
    FB_OP_SGESV, FB_OP_DGESV, FB_OP_CGESV, FB_OP_ZGESV,
    FB_OP_SPOSV, FB_OP_DPOSV, FB_OP_CPOSV, FB_OP_ZPOSV,
    FB_OP_SSYSV, FB_OP_DSYSV, FB_OP_CSYSV, FB_OP_ZSYSV,
    FB_OP_CHESV, FB_OP_ZHESV,
    FB_OP_SGETRF, FB_OP_DGETRF, FB_OP_CGETRF, FB_OP_ZGETRF,
    FB_OP_SPOTRF, FB_OP_DPOTRF, FB_OP_CPOTRF, FB_OP_ZPOTRF,
    FB_OP_SGETRI, FB_OP_DGETRI, FB_OP_CGETRI, FB_OP_ZGETRI,
    FB_OP_SPOTRI, FB_OP_DPOTRI, FB_OP_CPOTRI, FB_OP_ZPOTRI,
    FB_OP_STRTRI, FB_OP_DTRTRI, FB_OP_CTRTRI, FB_OP_ZTRTRI,
    FB_OP_SGEEV, FB_OP_DGEEV, FB_OP_CGEEV, FB_OP_ZGEEV,
    FB_OP_SSYEV, FB_OP_DSYEV,
    FB_OP_CHEEV, FB_OP_ZHEEV,
    FB_OP_SGESVD, FB_OP_DGESVD, FB_OP_CGESVD, FB_OP_ZGESVD,
    FB_OP_SGEQRF, FB_OP_DGEQRF, FB_OP_CGEQRF, FB_OP_ZGEQRF,
    FB_OP_SORGQR, FB_OP_DORGQR,
    FB_OP_CUNGQR, FB_OP_ZUNGQR,
    
    FB_OP_COUNT  /* Total: 212 operations */
} fb_operation_id_t;

/* Note: fb_dispatch_key_t and fb_dispatch_entry_t removed - dispatch is now
 * operation-specific with direct function pointers in vtable.
 * Old generic dispatch with fb_dtype_t is replaced with type-specific operations. */

/* ============================================================================
 * Dispatch Table Management
 * ========================================================================= */

/**
 * @brief Create a new dispatch table
 * 
 * @return New dispatch table, or NULL on failure
 */
fb_dispatch_table_t* fb_dispatch_table_create(void);

/**
 * @brief Initialize dispatch table from calibration data
 * 
 * Populates the dispatch table with optimal backend function pointers
 * based on calibration results.
 * 
 * @param table Dispatch table to initialize
 * @param calibration_db Calibration database
 * @return 0 on success, negative error code on failure
 */
int fb_dispatch_table_init_from_calibration(
    fb_dispatch_table_t *table,
    const struct fb_calibration_database_t *calibration_db
);

/**
 * @brief Free dispatch table
 * 
 * @param table Table to free
 */
void fb_dispatch_table_free(fb_dispatch_table_t *table);

/**
 * @brief Register a backend with the dispatch table
 * 
 * @param table Dispatch table
 * @param name Backend name
 * @param backend Backend vtable
 * @return 0 on success, negative error code on failure
 */
int fb_dispatch_register_backend(
    fb_dispatch_table_t *table,
    const char *name,
    const fb_backend_vtable_t *backend);

/**
 * @brief Auto-select best backend
 * 
 * @param table Dispatch table
 * @return 0 on success, negative error code on failure
 */
int fb_dispatch_auto_select(fb_dispatch_table_t *table);

/**
 * @brief Get active backend name
 * 
 * @param table Dispatch table
 * @return Backend name or NULL
 */
const char *fb_dispatch_get_active_backend_name(const fb_dispatch_table_t *table);

// COMMENTED OUT - uses old generic dispatch approach
// const fb_dispatch_entry_t* fb_dispatch_lookup(
//     const fb_dispatch_table_t *table,
//     const fb_dispatch_key_t *key
// );

// COMMENTED OUT - uses old generic dispatch approach
// int fb_dispatch_set(
//     fb_dispatch_table_t *table,
//     const fb_dispatch_key_t *key,
//     const fb_dispatch_entry_t *entry
// );


// COMMENTED OUT - uses old generic dispatch approach
// int fb_dispatch_table_rebuild(
//     fb_dispatch_table_t *table,
//     const fb_calibration_database_t *calibration_db,
//     const int *enabled_backends,
//     size_t backend_count
// );

/* ============================================================================
 * Backend Registry
 * ========================================================================= */

// COMMENTED OUT - uses fb_backend_t type that doesn't exist
// typedef struct {
//     int backend_id;
//     fb_backend_t *backend_handle;
//     fb_backend_info_t info;
//     fb_backend_vtable_t vtable;
//     bool enabled;
//     void *plugin_handle;
// } fb_registered_backend_t;

// COMMENTED OUT - uses fb_backend_t type that doesn't exist
// int fb_dispatch_register_backend(
//     int backend_id,
//     const fb_backend_vtable_t *vtable,
//     fb_backend_t *backend_handle
// );

// COMMENTED OUT - uses fb_registered_backend_t that doesn't exist
// int fb_dispatch_unregister_backend(int backend_id);

// COMMENTED OUT - uses fb_registered_backend_t that doesn't exist
// size_t fb_dispatch_get_registered_backends(
//     fb_registered_backend_t *backends,
//     size_t max_backends
// );

// COMMENTED OUT - legacy backend management functions
// int fb_dispatch_enable_backend(int backend_id);
// int fb_dispatch_disable_backend(int backend_id);
// bool fb_dispatch_is_backend_enabled(int backend_id);

/* ============================================================================
 * Thread-Local Context Management
 * ========================================================================= */

// COMMENTED OUT: fb_backend_context_t type doesn't exist in new type-specific architecture
// /**
//  * @brief Thread-local dispatch context
//  * 
//  * Each thread maintains its own context with backend handles.
//  */
// typedef struct {
//     fb_backend_context_t *backend_contexts[16];  /**< Per-backend contexts */
//     uint64_t call_count;                         /**< Number of BLAS calls */
//     uint64_t total_time_ns;                      /**< Total time in BLAS */
//     bool initialized;                            /**< Whether context is ready */
// } fb_thread_context_t;

// COMMENTED OUT: fb_thread_context_t doesn't exist in new architecture
// /**
//  * @brief Get or create thread-local context
//  * 
//  * This function is very fast after the first call (thread-local storage).
//  * 
//  * @return Thread context, or NULL on failure
//  */
// fb_thread_context_t* fb_dispatch_get_thread_context(void);

/**
 * @brief Initialize thread-local context
 * 
 * Should be called once per thread before any BLAS operations.
 * 
 * @return 0 on success, negative error code on failure
 */
int fb_dispatch_init_thread(void);

/**
 * @brief Cleanup thread-local context
 * 
 * Should be called when thread is done with BLAS operations.
 */
void fb_dispatch_cleanup_thread(void);

/* ============================================================================
 * Dispatch Utilities
 * ========================================================================= */

/* Commented out - uses old generic dispatch approach
 * @brief Create dispatch key from parameters
 * 
 * Helper function to construct a dispatch key from operation parameters.
 * 
 * @param key Output key
 * @param operation Operation ID
 * @param dtype Data type
 * @param layout Memory layout
 * @param m Dimension M
 * @param n Dimension N
 * @param k Dimension K
 *
void fb_dispatch_key_init(
    fb_dispatch_key_t *key,
    fb_operation_id_t operation,
    fb_dtype_t dtype,
    fb_layout_t layout,
    size_t m, size_t n, size_t k
);
*/

/* Commented out - uses old generic dispatch approach
 * @brief Hash function for dispatch keys
 * 
 * @param key Key to hash
 * @return Hash value
 *
uint64_t fb_dispatch_key_hash(const fb_dispatch_key_t *key);
*/

/* Commented out - uses old generic dispatch approach
 * @brief Compare two dispatch keys for equality
 * 
 * @param key1 First key
 * @param key2 Second key
 * @return true if equal, false otherwise
 *
bool fb_dispatch_key_equal(const fb_dispatch_key_t *key1, const fb_dispatch_key_t *key2);
*/

/**
 * @brief Get operation name from ID
 * 
 * @param operation Operation ID
 * @return String name, or NULL if invalid
 */
const char* fb_dispatch_get_operation_name(fb_operation_id_t operation);

/**
 * @brief Get operation ID from name
 * 
 * @param name Operation name (e.g., "gemm", "axpy")
 * @return Operation ID, or -1 if not found
 */
fb_operation_id_t fb_dispatch_get_operation_id(const char *name);

/* ============================================================================
 * Statistics Collection (Optional)
 * ========================================================================= */

/**
 * @brief Per-operation statistics
 */
typedef struct {
    uint64_t call_count;           /**< Number of calls */
    uint64_t total_time_ns;        /**< Total execution time */
    double min_time_ns;            /**< Fastest call */
    double max_time_ns;            /**< Slowest call */
    uint64_t total_flops;          /**< Total FLOPs computed */
    int backend_id;                /**< Which backend was used */
} fb_operation_stats_t;

/**
 * @brief Enable statistics collection
 * 
 * When enabled, tracks timing and usage for all operations.
 * Adds <1% overhead.
 * 
 * @param enable true to enable, false to disable
 */
void fb_dispatch_enable_statistics(bool enable);

/**
 * @brief Record statistics for an operation
 * 
 * Called internally after each operation when statistics are enabled.
 * 
 * @param operation Operation ID
 * @param backend_id Backend used
 * @param execution_time_ns Execution time
 * @param flops FLOPs computed
 */
void fb_dispatch_record_stats(
    fb_operation_id_t operation,
    int backend_id,
    double execution_time_ns,
    uint64_t flops
);

/**
 * @brief Get statistics for an operation
 * 
 * @param operation Operation ID
 * @param stats Output statistics
 * @return 0 on success, negative error code if no data
 */
int fb_dispatch_get_stats(fb_operation_id_t operation, fb_operation_stats_t *stats);

/**
 * @brief Reset all statistics
 */
void fb_dispatch_reset_stats(void);

/**
 * @brief Print statistics summary
 */
void fb_dispatch_print_stats(void);

/* ============================================================================
 * Global Dispatch System
 * ========================================================================= */

/**
 * @brief Initialize global dispatch system
 * 
 * @return 0 on success, negative error code on failure
 */
int fb_dispatch_init_global(void);

/**
 * @brief Get global dispatch table
 * 
 * @return Global dispatch table
 */
fb_dispatch_table_t* fb_dispatch_global(void);

/**
 * @brief Cleanup global dispatch system
 */
void fb_dispatch_finalize_global(void);

#ifdef __cplusplus
}
#endif

#endif /* FB_DISPATCH_H */
