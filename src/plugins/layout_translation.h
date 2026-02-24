/**
 * @file layout_translation.h
 * @brief Layout translation utilities for row-major ↔ column-major conversion
 * 
 * GPU backends (cuBLAS, rocBLAS, CLBlast) are inherently column-major (Fortran convention).
 * CPU backends (BLAS, AOCL-BLIS) support both layouts via CBLAS interface.
 * 
 * This module provides automatic translation for backends that don't support both layouts.
 */

#ifndef FASTER_BLASTER_LAYOUT_TRANSLATION_H
#define FASTER_BLASTER_LAYOUT_TRANSLATION_H

#include "../backends/backend_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Mathematical identity for layout conversion:
 * 
 * Row-Major: C = op(A) * op(B)
 * where C[m,n], A[m,k], B[k,n]
 * 
 * Is equivalent to:
 * 
 * Column-Major: C^T = op(B)^T * op(A)^T
 * where C^T[n,m], B^T[n,k], A^T[k,m]
 * 
 * For GEMM this means swapping:
 * - A ↔ B (matrix operands)
 * - m ↔ n (dimensions)
 * - lda ↔ ldb (leading dimensions)
 * - transa ↔ transb (transpose flags)
 * 
 * For GEMV (y = op(A)*x):
 * Row-Major A[m,n] with trans → Column-Major requires:
 * - If trans='N': swap m↔n, transpose becomes 'T'
 * - If trans='T': swap m↔n, transpose becomes 'N'
 */

/**
 * Translate GEMM from row-major to column-major
 * 
 * Input: Row-major C = alpha*op(A)*op(B) + beta*C
 * Output: Column-major parameters for C^T = alpha*op(B)^T*op(A)^T + beta*C^T
 */
static inline void translate_gemm_layout(
    fb_layout_t* layout_inout,
    fb_transpose_t* transa_inout, fb_transpose_t* transb_inout,
    int64_t* m_inout, int64_t* n_inout, int64_t* k,
    const float** a_inout, int64_t* lda_inout,
    const float** b_inout, int64_t* ldb_inout,
    float** c_inout, int64_t* ldc_inout)
{
    if (*layout_inout == FB_LAYOUT_ROW_MAJOR) {
        /* Swap operands and dimensions */
        const float* tmp_mat = *a_inout;
        *a_inout = *b_inout;
        *b_inout = tmp_mat;
        
        fb_transpose_t tmp_trans = *transa_inout;
        *transa_inout = *transb_inout;
        *transb_inout = tmp_trans;
        
        int64_t tmp_dim = *m_inout;
        *m_inout = *n_inout;
        *n_inout = tmp_dim;
        
        int64_t tmp_ld = *lda_inout;
        *lda_inout = *ldb_inout;
        *ldb_inout = tmp_ld;
        
        /* ldc needs to be adjusted: was n (columns), now m (rows in col-major) */
        *ldc_inout = *m_inout;
        
        *layout_inout = FB_LAYOUT_COL_MAJOR;
    }
}

/**
 * Translate GEMV from row-major to column-major
 * 
 * Input: Row-major y = alpha*op(A)*x + beta*y
 * Output: Column-major parameters
 */
static inline void translate_gemv_layout(
    fb_layout_t* layout_inout,
    fb_transpose_t* trans_inout,
    int64_t* m_inout, int64_t* n_inout,
    const float** a_inout, int64_t* lda_inout)
{
    if (*layout_inout == FB_LAYOUT_ROW_MAJOR) {
        /* For GEMV, transpose the operation */
        int64_t tmp = *m_inout;
        *m_inout = *n_inout;
        *n_inout = tmp;
        
        /* Flip transpose flag */
        if (*trans_inout == FB_NO_TRANS) {
            *trans_inout = FB_TRANS;
        } else if (*trans_inout == FB_TRANS) {
            *trans_inout = FB_NO_TRANS;
        }
        /* CONJ_TRANS stays as CONJ_TRANS */
        
        /* lda becomes the other dimension */
        *lda_inout = *m_inout;
        
        *layout_inout = FB_LAYOUT_COL_MAJOR;
    }
}

/**
 * Double-precision variants
 */
static inline void translate_gemm_layout_double(
    fb_layout_t* layout_inout,
    fb_transpose_t* transa_inout, fb_transpose_t* transb_inout,
    int64_t* m_inout, int64_t* n_inout, int64_t* k,
    const double** a_inout, int64_t* lda_inout,
    const double** b_inout, int64_t* ldb_inout,
    double** c_inout, int64_t* ldc_inout)
{
    if (*layout_inout == FB_LAYOUT_ROW_MAJOR) {
        const double* tmp_mat = *a_inout;
        *a_inout = *b_inout;
        *b_inout = tmp_mat;
        
        fb_transpose_t tmp_trans = *transa_inout;
        *transa_inout = *transb_inout;
        *transb_inout = tmp_trans;
        
        int64_t tmp_dim = *m_inout;
        *m_inout = *n_inout;
        *n_inout = tmp_dim;
        
        int64_t tmp_ld = *lda_inout;
        *lda_inout = *ldb_inout;
        *ldb_inout = tmp_ld;
        
        *ldc_inout = *m_inout;
        *layout_inout = FB_LAYOUT_COL_MAJOR;
    }
}

static inline void translate_gemv_layout_double(
    fb_layout_t* layout_inout,
    fb_transpose_t* trans_inout,
    int64_t* m_inout, int64_t* n_inout,
    const double** a_inout, int64_t* lda_inout)
{
    if (*layout_inout == FB_LAYOUT_ROW_MAJOR) {
        int64_t tmp = *m_inout;
        *m_inout = *n_inout;
        *n_inout = tmp;
        
        if (*trans_inout == FB_NO_TRANS) {
            *trans_inout = FB_TRANS;
        } else if (*trans_inout == FB_TRANS) {
            *trans_inout = FB_NO_TRANS;
        }
        
        *lda_inout = *m_inout;
        *layout_inout = FB_LAYOUT_COL_MAJOR;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_LAYOUT_TRANSLATION_H */
