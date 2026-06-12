/**
 * @file dispatch_api.h
 * @brief User-facing dispatch API - automatic and explicit device selection
 * 
 * This is the primary interface for users of the faster-blaster library.
 * It provides two dispatch modes:
 * 
 * 1. AUTOMATIC DISPATCH: Library intelligently selects best device
 *    - fb_sgemm_auto(), fb_dgetrf_auto(), etc.
 *    - Smart load balancing, data locality, power management
 *    - User just calls function, library handles everything
 * 
 * 2. EXPLICIT DISPATCH: User specifies exact device
 *    - fb_sgemm_on_device(device_id, ...)
 *    - Full control when needed
 *    - Still benefits from backend abstraction
 * 
 * Both modes use the same underlying backend traits (CPU/GPU vtables).
 */

#ifndef FASTER_BLASTER_DISPATCH_API_H
#define FASTER_BLASTER_DISPATCH_API_H

#include "compute_device.h"
#include "compute_manager.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Library Initialization
 * ========================================================================== */

/**
 * @brief Initialize faster-blaster library
 * 
 * Discovers all devices, initializes backends, loads calibration data.
 * Must be called before any other library functions.
 * 
 * @param config Manager configuration (NULL for defaults)
 * @return 0 on success, non-zero on error
 */
int fb_init(const fb_manager_config_t* config);

/**
 * @brief Shutdown library and release all resources
 */
void fb_shutdown(void);

/**
 * @brief Check if library is initialized
 * @return true if initialized
 */
bool fb_is_initialized(void);

/**
 * @brief Get library version string
 * @return Version string (e.g., "1.0.0")
 */
const char* fb_get_version(void);

/* ============================================================================
 * Device Management API
 * ========================================================================== */

/**
 * @brief Get number of available devices
 * @return Total device count
 */
int fb_get_device_count(void);

/**
 * @brief Get device information
 * @param device_id Device ID
 * @return Device handle, or NULL if invalid
 */
const fb_compute_device_t* fb_get_device_info(int device_id);

/**
 * @brief Print all available devices
 */
void fb_print_devices(void);

/**
 * @brief Set default device for operations
 * @param device_id Device ID to use as default
 * @return 0 on success, non-zero on error
 */
int fb_set_default_device(int device_id);

/**
 * @brief Get current default device
 * @return Default device ID, or -1 if using automatic selection
 */
int fb_get_default_device(void);

/* ============================================================================
 * Policy Control
 * ========================================================================== */

/**
 * @brief Set global dispatch policy
 * @param policy Scheduling policy to use
 * @return 0 on success, non-zero on error
 */
int fb_set_dispatch_policy(fb_scheduling_policy_t policy);

/**
 * @brief Get current dispatch policy
 * @return Current policy
 */
fb_scheduling_policy_t fb_get_dispatch_policy(void);

/**
 * @brief Set scoring weights for multi-factor policy
 * @param speed_weight Performance weight (0-1)
 * @param load_weight Load balance weight (0-1)
 * @param locality_weight Data locality weight (0-1)
 * @param power_weight Power efficiency weight (0-1)
 */
void fb_set_scoring_weights(double speed_weight,
                            double load_weight,
                            double locality_weight,
                            double power_weight);

/* ============================================================================
 * Data Locality Hints
 * ========================================================================== */

/**
 * @brief Register data location for locality-aware dispatch
 * @param ptr Data pointer
 * @param device_id Device where data resides
 * @param size Data size in bytes
 */
void fb_register_data_location(const void* ptr, int device_id, size_t size);

/**
 * @brief Unregister data location
 * @param ptr Data pointer
 */
void fb_unregister_data_location(const void* ptr);

/**
 * @brief Query where data is located
 * @param ptr Data pointer
 * @return Device ID, or -1 if unknown
 */
int fb_query_data_location(const void* ptr);

/* ============================================================================
 * BLAS Level 1: Automatic Dispatch
 * ========================================================================== */

/**
 * @brief SAXPY with automatic device selection
 * @param n Vector length
 * @param alpha Scalar alpha
 * @param x Input vector X
 * @param incx Stride for X
 * @param y Input/output vector Y
 * @param incy Stride for Y
 */
void fb_saxpy_auto(int n, float alpha, const float* x, int incx, float* y, int incy);

void fb_daxpy_auto(int n, double alpha, const double* x, int incx, double* y, int incy);

/**
 * @brief SGEMV with automatic device selection
 * @param trans Transpose operation ('N', 'T', 'C')
 * @param m Number of rows
 * @param n Number of columns
 * @param alpha Scalar alpha
 * @param a Matrix A
 * @param lda Leading dimension of A
 * @param x Vector X
 * @param incx Stride for X
 * @param beta Scalar beta
 * @param y Vector Y
 * @param incy Stride for Y
 */
void fb_sgemv_auto(char trans, int m, int n, float alpha,
                   const float* a, int lda, const float* x, int incx,
                   float beta, float* y, int incy);

void fb_dgemv_auto(char trans, int m, int n, double alpha,
                   const double* a, int lda, const double* x, int incx,
                   double beta, double* y, int incy);

/* ============================================================================
 * BLAS Level 3: Automatic Dispatch
 * ========================================================================== */

/**
 * @brief SGEMM with automatic device selection
 * 
 * This is the flagship function - demonstrates hybrid dispatch in action.
 * The library will:
 * 1. Estimate problem size and complexity
 * 2. Check current device loads
 * 3. Consider data location
 * 4. Select optimal device (CPU or GPU)
 * 5. Execute using appropriate backend
 * 
 * @param transa Transpose A ('N' or 'T')
 * @param transb Transpose B ('N' or 'T')
 * @param m Number of rows of A and C
 * @param n Number of columns of B and C
 * @param k Number of columns of A and rows of B
 * @param alpha Scalar alpha
 * @param a Matrix A
 * @param lda Leading dimension of A
 * @param b Matrix B
 * @param ldb Leading dimension of B
 * @param beta Scalar beta
 * @param c Matrix C (input/output)
 * @param ldc Leading dimension of C
 */
void fb_sgemm_auto(char transa, char transb, int m, int n, int k,
                   float alpha, const float* a, int lda,
                   const float* b, int ldb, float beta,
                   float* c, int ldc);

void fb_dgemm_auto(char transa, char transb, int m, int n, int k,
                   double alpha, const double* a, int lda,
                   const double* b, int ldb, double beta,
                   double* c, int ldc);

void fb_cgemm_auto(char transa, char transb, int m, int n, int k,
                   const void* alpha, const void* a, int lda,
                   const void* b, int ldb, const void* beta,
                   void* c, int ldc);

void fb_zgemm_auto(char transa, char transb, int m, int n, int k,
                   const void* alpha, const void* a, int lda,
                   const void* b, int ldb, const void* beta,
                   void* c, int ldc);

/**
 * @brief SSYMM with automatic device selection
 */
void fb_ssymm_auto(char side, char uplo, int m, int n,
                   float alpha, const float* a, int lda,
                   const float* b, int ldb, float beta,
                   float* c, int ldc);

void fb_dsymm_auto(char side, char uplo, int m, int n,
                   double alpha, const double* a, int lda,
                   const double* b, int ldb, double beta,
                   double* c, int ldc);

/**
 * @brief STRSM with automatic device selection
 */
void fb_strsm_auto(char side, char uplo, char transa, char diag,
                   int m, int n, float alpha, const float* a, int lda,
                   float* b, int ldb);

void fb_dtrsm_auto(char side, char uplo, char transa, char diag,
                   int m, int n, double alpha, const double* a, int lda,
                   double* b, int ldb);

/* ============================================================================
 * LAPACK: Automatic Dispatch
 * ========================================================================== */

/**
 * @brief SGETRF (LU factorization) with automatic device selection
 * @param m Number of rows
 * @param n Number of columns
 * @param a Matrix A (input/output)
 * @param lda Leading dimension
 * @param ipiv Pivot indices
 * @return Info code (0 = success)
 */
int fb_sgetrf_auto(int m, int n, float* a, int lda, int* ipiv);

int fb_dgetrf_auto(int m, int n, double* a, int lda, int* ipiv);

/**
 * @brief SPOTRF (Cholesky factorization) with automatic device selection
 * @param uplo Upper or lower triangle ('U' or 'L')
 * @param n Matrix dimension
 * @param a Matrix A (input/output)
 * @param lda Leading dimension
 * @return Info code (0 = success)
 */
int fb_spotrf_auto(char uplo, int n, float* a, int lda);

int fb_dpotrf_auto(char uplo, int n, double* a, int lda);

/**
 * @brief SGEQRF (QR factorization) with automatic device selection
 * @param m Number of rows
 * @param n Number of columns
 * @param a Matrix A (input/output)
 * @param lda Leading dimension
 * @param tau Scalar factors
 * @return Info code (0 = success)
 */
int fb_sgeqrf_auto(int m, int n, float* a, int lda, float* tau);

int fb_dgeqrf_auto(int m, int n, double* a, int lda, double* tau);

/**
 * @brief SGESVD (SVD) with automatic device selection
 * @param jobu Compute U ('A', 'S', 'O', 'N')
 * @param jobvt Compute VT ('A', 'S', 'O', 'N')
 * @param m Number of rows
 * @param n Number of columns
 * @param a Matrix A (input/output)
 * @param lda Leading dimension of A
 * @param s Singular values
 * @param u Matrix U
 * @param ldu Leading dimension of U
 * @param vt Matrix VT
 * @param ldvt Leading dimension of VT
 * @return Info code (0 = success)
 */
int fb_sgesvd_auto(char jobu, char jobvt, int m, int n,
                   float* a, int lda, float* s,
                   float* u, int ldu, float* vt, int ldvt);

int fb_dgesvd_auto(char jobu, char jobvt, int m, int n,
                   double* a, int lda, double* s,
                   double* u, int ldu, double* vt, int ldvt);

/* ============================================================================
 * Explicit Device Dispatch
 * ========================================================================== */

/**
 * @brief SGEMM on specific device
 * @param device_id Device to execute on
 * @param ... Same parameters as fb_sgemm_auto
 */
void fb_sgemm_on_device(int device_id, char transa, char transb,
                        int m, int n, int k, float alpha,
                        const float* a, int lda, const float* b, int ldb,
                        float beta, float* c, int ldc);

void fb_dgemm_on_device(int device_id, char transa, char transb,
                        int m, int n, int k, double alpha,
                        const double* a, int lda, const double* b, int ldb,
                        double beta, double* c, int ldc);

/**
 * @brief SGETRF on specific device
 */
int fb_sgetrf_on_device(int device_id, int m, int n, float* a, int lda, int* ipiv);

int fb_dgetrf_on_device(int device_id, int m, int n, double* a, int lda, int* ipiv);

/* ============================================================================
 * Batched Operations: Automatic Dispatch
 * ========================================================================== */

/**
 * @brief Batched SGEMM with automatic device selection
 * 
 * Batched operations are particularly efficient on GPUs but also work on CPUs.
 * The library will intelligently choose based on batch size and problem dimensions.
 * 
 * @param transa Transpose A
 * @param transb Transpose B
 * @param m Number of rows
 * @param n Number of columns
 * @param k Inner dimension
 * @param alpha Scalar alpha
 * @param a_array Array of matrix A pointers
 * @param lda Leading dimension of A
 * @param b_array Array of matrix B pointers
 * @param ldb Leading dimension of B
 * @param beta Scalar beta
 * @param c_array Array of matrix C pointers
 * @param ldc Leading dimension of C
 * @param batch_count Number of matrices in batch
 * @return 0 on success, non-zero on error
 */
int fb_sgemm_batched_auto(char transa, char transb, int m, int n, int k,
                          float alpha, const float** a_array, int lda,
                          const float** b_array, int ldb, float beta,
                          float** c_array, int ldc, int batch_count);

int fb_dgemm_batched_auto(char transa, char transb, int m, int n, int k,
                          double alpha, const double** a_array, int lda,
                          const double** b_array, int ldb, double beta,
                          double** c_array, int ldc, int batch_count);

/* ============================================================================
 * Asynchronous Operations
 * ========================================================================== */

/**
 * @brief Async operation handle
 */
typedef struct fb_async_handle fb_async_handle_t;

/**
 * @brief SGEMM async with automatic device selection
 * 
 * Returns immediately, operation executes in background.
 * 
 * @param ... Same parameters as fb_sgemm_auto
 * @return Async handle for wait/query
 */
fb_async_handle_t* fb_sgemm_auto_async(char transa, char transb,
                                       int m, int n, int k, float alpha,
                                       const float* a, int lda,
                                       const float* b, int ldb, float beta,
                                       float* c, int ldc);

/**
 * @brief Wait for async operation to complete
 * @param handle Async handle
 * @return 0 on success, non-zero on error
 */
int fb_async_wait(fb_async_handle_t* handle);

/**
 * @brief Check if async operation is complete
 * @param handle Async handle
 * @return true if complete
 */
bool fb_async_is_complete(fb_async_handle_t* handle);

/**
 * @brief Destroy async handle
 * @param handle Handle to destroy
 */
void fb_async_destroy(fb_async_handle_t* handle);

/* ============================================================================
 * Statistics and Monitoring
 * ========================================================================== */

/**
 * @brief Get dispatch statistics
 * @param stats Output: statistics
 */
void fb_get_stats(fb_manager_stats_t* stats);

/**
 * @brief Reset statistics counters
 */
void fb_reset_stats(void);

/**
 * @brief Print library status
 */
void fb_print_status(void);

/**
 * @brief Enable verbose logging
 * @param enable true to enable verbose output
 */
void fb_set_verbose(bool enable);

/**
 * @brief Get last device selection explanation
 * @param explanation Output: explanation
 * @return 0 on success, non-zero if not available
 */
int fb_get_last_selection_explanation(fb_selection_explanation_t* explanation);

/* ============================================================================
 * Performance Tuning
 * ========================================================================== */

/**
 * @brief Calibrate all devices
 * @param quick_mode If true, fast calibration; if false, thorough calibration
 * @return 0 on success, non-zero on error
 */
int fb_calibrate_all_devices(bool quick_mode);

/**
 * @brief Save calibration data
 * @param filename Output filename
 * @return 0 on success, non-zero on error
 */
int fb_save_calibration(const char* filename);

/**
 * @brief Load calibration data
 * @param filename Input filename
 * @return 0 on success, non-zero on error
 */
int fb_load_calibration(const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_DISPATCH_API_H */
