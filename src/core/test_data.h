/**
 * @file test_data.h
 * @brief Test data generation for calibration
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_TEST_DATA_H
#define FB_TEST_DATA_H

#include "../backends/backend_interface.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Test matrix properties
 */
typedef enum {
    FB_TEST_RANDOM,          /**< Random values in [0, 1] */
    FB_TEST_IDENTITY,        /**< Identity matrix */
    FB_TEST_ZEROS,           /**< All zeros */
    FB_TEST_ONES,            /**< All ones */
    FB_TEST_DIAGONAL,        /**< Diagonal matrix */
    FB_TEST_SYMMETRIC,       /**< Symmetric matrix */
    FB_TEST_TRIANGULAR,      /**< Triangular matrix */
    FB_TEST_ILL_CONDITIONED  /**< Ill-conditioned (high condition number) */
} fb_test_matrix_type_t;

// COMMENTED OUT: All functions use obsolete fb_dtype_t type
// These will need to be redesigned for type-specific architecture
/*
int fb_test_generate_matrix(
    void *matrix,
    size_t rows,
    size_t cols,
    fb_dtype_t dtype,
    fb_test_matrix_type_t type,
    uint32_t seed
);

int fb_test_generate_vector(
    void *vector,
    size_t length,
    fb_dtype_t dtype,
    fb_test_matrix_type_t type,
    uint32_t seed
);

bool fb_test_compare_arrays(
    const void *expected,
    const void *actual,
    size_t length,
    fb_dtype_t dtype,
    double tolerance,
    double *max_error,
    double *avg_error
);

double fb_test_condition_number(
    const void *matrix,
    size_t rows,
    size_t cols,
    fb_dtype_t dtype
);

size_t fb_dtype_size(fb_dtype_t dtype);
*/

/**
 * @brief Initialize random number generator
 * 
 * @param seed Random seed
 */
void fb_test_init_rng(uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif /* FB_TEST_DATA_H */
