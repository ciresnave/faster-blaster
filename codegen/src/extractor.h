/**
 * Function signature extraction using libclang
 */

#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <stddef.h>

/**
 * Represents a single extracted function/operation
 */
typedef struct {
    char name[256];              /* Normalized operation name (saxpy) */
    char actual_name[256];       /* Real function name (cblas_saxpy or saxpy_) */
    char return_type[128];       /* Return type (void, float, double, etc.) */
    char** parameters;           /* Parameter list as strings */
    int param_count;
    char category[64];           /* blas_level1, blas_level2, blas_level3, lapack_aux, lapack_comp, lapack_driver */
    char backend[64];            /* Backend name (openblas, reference, etc.) */
} Operation;

/**
 * List of extracted operations
 */
typedef struct {
    Operation* operations;
    int count;
    int capacity;
} OperationList;

/**
 * Extract all BLAS/LAPACK function signatures from a header file
 * 
 * Args:
 *   header_path: Path to C header file
 *   backend_name: Name of backend (openblas, mkl, etc.)
 *   convention: "c" for cblas_*, "fortran" for *_
 * 
 * Returns: Allocated OperationList or NULL on failure
 */
OperationList* extract_operations_from_header(
    const char* header_path,
    const char* backend_name,
    const char* convention
);

/**
 * Free extracted operations list
 */
void free_operations(OperationList* ops);

#endif
