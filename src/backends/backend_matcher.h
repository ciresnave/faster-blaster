/**
 * @file backend_matcher.h
 * @brief Device-to-Backend Matching System Header
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_BACKEND_MATCHER_H
#define FASTER_BLASTER_BACKEND_MATCHER_H

#include "backend_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Backend instance handle (opaque)
 */
typedef struct backend_instance backend_instance_t;
typedef backend_instance_t fb_backend_instance_t;  /* Public alias */

/**
 * Initialize backend matcher system
 * @return 0 on success, negative on error
 */
int fb_backend_matcher_init(void);

/**
 * Shutdown backend matcher and cleanup all instances
 */
void fb_backend_matcher_shutdown(void);

/**
 * Get or create backend instance for a specific device
 * This function automatically selects the best backend for the device
 * 
 * @param device_id Device ID from device registry
 * @return Backend instance pointer, or NULL on error
 */
fb_backend_instance_t* fb_backend_get_for_device(int device_id);

/**
 * Get backend information for a device
 * 
 * @param device_id Device ID
 * @param info Output parameter for backend metadata
 * @return 0 on success, negative on error
 */
int fb_backend_get_device_backend_info(int device_id, fb_backend_metadata_t* info);

/**
 * Execute SGEMM on appropriate backend for device
 * 
 * @param device_id Target device
 * @param transa Transpose A ('N' or 'T')
 * @param transb Transpose B ('N' or 'T')
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param alpha Scalar alpha
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param B Matrix B
 * @param ldb Leading dimension of B
 * @param beta Scalar beta
 * @param C Matrix C (output)
 * @param ldc Leading dimension of C
 * @return 0 on success, negative on error
 */
int fb_backend_execute_sgemm(int device_id, char transa, char transb,
                              int m, int n, int k, float alpha,
                              const float* A, int lda, const float* B, int ldb,
                              float beta, float* C, int ldc);

/**
 * Execute DGEMM on appropriate backend for device
 * @see fb_backend_execute_sgemm for parameter documentation
 */
int fb_backend_execute_dgemm(int device_id, char transa, char transb,
                              int m, int n, int k, double alpha,
                              const double* A, int lda, const double* B, int ldb,
                              double beta, double* C, int ldc);

/**
 * Print backend configuration for all devices
 */
void fb_backend_print_configuration(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_BACKEND_MATCHER_H */
