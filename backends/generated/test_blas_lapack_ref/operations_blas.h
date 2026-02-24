/**
 * GENERATED FILE - DO NOT EDIT
 * Backend: blas
 * Operations: 128
 * Categories:
 *   BLAS Level 1: 26
 *   BLAS Level 2: 42
 *   BLAS Level 3: 32
 *   LAPACK Auxiliary: 0
 *   LAPACK Computational: 28
 *   LAPACK Driver: 0
 */

#ifndef FB_OPERATIONS_blas_H
#define FB_OPERATIONS_blas_H

/* blas_level1: 26 operations */
#ifdef OP
OP(
    caxpy_ref,  /* normalized name */
    void,  /* return type */
    caxpy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ccopy_ref,  /* normalized name */
    void,  /* return type */
    ccopy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cnrm2_ref,  /* normalized name */
    void,  /* return type */
    cnrm2_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cscal_ref,  /* normalized name */
    void,  /* return type */
    cscal_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cswap_ref,  /* normalized name */
    void,  /* return type */
    cswap_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dasum_ref,  /* normalized name */
    void,  /* return type */
    dasum_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    daxpy_ref,  /* normalized name */
    void,  /* return type */
    daxpy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dcopy_ref,  /* normalized name */
    void,  /* return type */
    dcopy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ddot_ref,  /* normalized name */
    void,  /* return type */
    ddot_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dnrm2_ref,  /* normalized name */
    void,  /* return type */
    dnrm2_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    drot_ref,  /* normalized name */
    void,  /* return type */
    drot_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dscal_ref,  /* normalized name */
    void,  /* return type */
    dscal_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dswap_ref,  /* normalized name */
    void,  /* return type */
    dswap_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sasum_ref,  /* normalized name */
    void,  /* return type */
    sasum_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    saxpy_ref,  /* normalized name */
    void,  /* return type */
    saxpy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    scopy_ref,  /* normalized name */
    void,  /* return type */
    scopy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sdot_ref,  /* normalized name */
    void,  /* return type */
    sdot_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    snrm2_ref,  /* normalized name */
    void,  /* return type */
    snrm2_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    srot_ref,  /* normalized name */
    void,  /* return type */
    srot_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sscal_ref,  /* normalized name */
    void,  /* return type */
    sscal_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sswap_ref,  /* normalized name */
    void,  /* return type */
    sswap_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zaxpy_ref,  /* normalized name */
    void,  /* return type */
    zaxpy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zcopy_ref,  /* normalized name */
    void,  /* return type */
    zcopy_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    znrm2_ref,  /* normalized name */
    void,  /* return type */
    znrm2_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zscal_ref,  /* normalized name */
    void,  /* return type */
    zscal_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zswap_ref,  /* normalized name */
    void,  /* return type */
    zswap_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
#undef OP

/* blas_level2: 42 operations */
#ifdef OP
OP(
    cgbmv_,  /* normalized name */
    void,  /* return type */
    cgbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cher_,  /* normalized name */
    void,  /* return type */
    cher_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cher2_,  /* normalized name */
    void,  /* return type */
    cher2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cspr_,  /* normalized name */
    void,  /* return type */
    cspr_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cspr2_,  /* normalized name */
    void,  /* return type */
    cspr2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dgbmv_ref,  /* normalized name */
    void,  /* return type */
    dgbmv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dgemv_ref,  /* normalized name */
    void,  /* return type */
    dgemv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dgemv_batched_,  /* normalized name */
    void,  /* return type */
    dgemv_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dger_ref,  /* normalized name */
    void,  /* return type */
    dger_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dher_,  /* normalized name */
    void,  /* return type */
    dher_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dher2_,  /* normalized name */
    void,  /* return type */
    dher2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dsbmv_,  /* normalized name */
    void,  /* return type */
    dsbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dspr_,  /* normalized name */
    void,  /* return type */
    dspr_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dspr2_,  /* normalized name */
    void,  /* return type */
    dspr2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dsymv_ref,  /* normalized name */
    void,  /* return type */
    dsymv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrmv_ref,  /* normalized name */
    void,  /* return type */
    dtrmv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrmv_batched_,  /* normalized name */
    void,  /* return type */
    dtrmv_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrsv_ref,  /* normalized name */
    void,  /* return type */
    dtrsv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrsv_batched_,  /* normalized name */
    void,  /* return type */
    dtrsv_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sgbmv_ref,  /* normalized name */
    void,  /* return type */
    sgbmv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sgemv_ref,  /* normalized name */
    void,  /* return type */
    sgemv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sgemv_batched_,  /* normalized name */
    void,  /* return type */
    sgemv_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sger_ref,  /* normalized name */
    void,  /* return type */
    sger_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sher_,  /* normalized name */
    void,  /* return type */
    sher_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sher2_,  /* normalized name */
    void,  /* return type */
    sher2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ssbmv_,  /* normalized name */
    void,  /* return type */
    ssbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sspr_,  /* normalized name */
    void,  /* return type */
    sspr_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sspr2_,  /* normalized name */
    void,  /* return type */
    sspr2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ssymv_ref,  /* normalized name */
    void,  /* return type */
    ssymv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strmv_ref,  /* normalized name */
    void,  /* return type */
    strmv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strmv_batched_,  /* normalized name */
    void,  /* return type */
    strmv_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strsv_ref,  /* normalized name */
    void,  /* return type */
    strsv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strsv_batched_,  /* normalized name */
    void,  /* return type */
    strsv_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zgbmv_,  /* normalized name */
    void,  /* return type */
    zgbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zher_,  /* normalized name */
    void,  /* return type */
    zher_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zher2_,  /* normalized name */
    void,  /* return type */
    zher2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zspr_,  /* normalized name */
    void,  /* return type */
    zspr_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zspr2_,  /* normalized name */
    void,  /* return type */
    zspr2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cher2k_,  /* normalized name */
    void,  /* return type */
    cher2k_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cherk_,  /* normalized name */
    void,  /* return type */
    cherk_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zher2k_,  /* normalized name */
    void,  /* return type */
    zher2k_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zherk_,  /* normalized name */
    void,  /* return type */
    zherk_,  /* actual function */
    (void)  /* parameters (simplified) */
)
#undef OP

/* blas_level3: 32 operations */
#ifdef OP
OP(
    dgemm_batched_,  /* normalized name */
    void,  /* return type */
    dgemm_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrmm_batched_,  /* normalized name */
    void,  /* return type */
    dtrmm_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrsm_batched_,  /* normalized name */
    void,  /* return type */
    dtrsm_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sgemm_batched_,  /* normalized name */
    void,  /* return type */
    sgemm_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strmm_batched_,  /* normalized name */
    void,  /* return type */
    strmm_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strsm_batched_,  /* normalized name */
    void,  /* return type */
    strsm_batched_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    cgemm_ref,  /* normalized name */
    void,  /* return type */
    cgemm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    chemm_,  /* normalized name */
    void,  /* return type */
    chemm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    csymm_,  /* normalized name */
    void,  /* return type */
    csymm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    csyr2k_,  /* normalized name */
    void,  /* return type */
    csyr2k_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    csyrk_,  /* normalized name */
    void,  /* return type */
    csyrk_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ctrmm_,  /* normalized name */
    void,  /* return type */
    ctrmm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ctrsm_,  /* normalized name */
    void,  /* return type */
    ctrsm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dgemm_ref,  /* normalized name */
    void,  /* return type */
    dgemm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dsymm_ref,  /* normalized name */
    void,  /* return type */
    dsymm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dsyr2k_ref,  /* normalized name */
    void,  /* return type */
    dsyr2k_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dsyrk_ref,  /* normalized name */
    void,  /* return type */
    dsyrk_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrmm_ref,  /* normalized name */
    void,  /* return type */
    dtrmm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtrsm_ref,  /* normalized name */
    void,  /* return type */
    dtrsm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    sgemm_ref,  /* normalized name */
    void,  /* return type */
    sgemm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ssymm_ref,  /* normalized name */
    void,  /* return type */
    ssymm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ssyr2k_ref,  /* normalized name */
    void,  /* return type */
    ssyr2k_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ssyrk_ref,  /* normalized name */
    void,  /* return type */
    ssyrk_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strmm_ref,  /* normalized name */
    void,  /* return type */
    strmm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    strsm_ref,  /* normalized name */
    void,  /* return type */
    strsm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zgemm_ref,  /* normalized name */
    void,  /* return type */
    zgemm_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zhemm_,  /* normalized name */
    void,  /* return type */
    zhemm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zsymm_,  /* normalized name */
    void,  /* return type */
    zsymm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zsyr2k_,  /* normalized name */
    void,  /* return type */
    zsyr2k_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zsyrk_,  /* normalized name */
    void,  /* return type */
    zsyrk_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ztrmm_,  /* normalized name */
    void,  /* return type */
    ztrmm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ztrsm_,  /* normalized name */
    void,  /* return type */
    ztrsm_,  /* actual function */
    (void)  /* parameters (simplified) */
)
#undef OP

/* lapack_computational: 28 operations */
#ifdef OP
OP(
    chbmv_,  /* normalized name */
    void,  /* return type */
    chbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    chemv_,  /* normalized name */
    void,  /* return type */
    chemv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    csyr_,  /* normalized name */
    void,  /* return type */
    csyr_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    csyr2_,  /* normalized name */
    void,  /* return type */
    csyr2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ctbmv_,  /* normalized name */
    void,  /* return type */
    ctbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ctbsv_,  /* normalized name */
    void,  /* return type */
    ctbsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ctpsv_,  /* normalized name */
    void,  /* return type */
    ctpsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dhbmv_,  /* normalized name */
    void,  /* return type */
    dhbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dhemv_,  /* normalized name */
    void,  /* return type */
    dhemv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dsyr_ref,  /* normalized name */
    void,  /* return type */
    dsyr_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dsyr2_ref,  /* normalized name */
    void,  /* return type */
    dsyr2_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtbmv_ref,  /* normalized name */
    void,  /* return type */
    dtbmv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtbsv_,  /* normalized name */
    void,  /* return type */
    dtbsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    dtpsv_,  /* normalized name */
    void,  /* return type */
    dtpsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    shbmv_,  /* normalized name */
    void,  /* return type */
    shbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    shemv_,  /* normalized name */
    void,  /* return type */
    shemv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ssyr_ref,  /* normalized name */
    void,  /* return type */
    ssyr_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ssyr2_ref,  /* normalized name */
    void,  /* return type */
    ssyr2_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    stbmv_ref,  /* normalized name */
    void,  /* return type */
    stbmv_ref,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    stbsv_,  /* normalized name */
    void,  /* return type */
    stbsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    stpsv_,  /* normalized name */
    void,  /* return type */
    stpsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zhbmv_,  /* normalized name */
    void,  /* return type */
    zhbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zhemv_,  /* normalized name */
    void,  /* return type */
    zhemv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zsyr_,  /* normalized name */
    void,  /* return type */
    zsyr_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    zsyr2_,  /* normalized name */
    void,  /* return type */
    zsyr2_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ztbmv_,  /* normalized name */
    void,  /* return type */
    ztbmv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ztbsv_,  /* normalized name */
    void,  /* return type */
    ztbsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
OP(
    ztpsv_,  /* normalized name */
    void,  /* return type */
    ztpsv_,  /* actual function */
    (void)  /* parameters (simplified) */
)
#undef OP

#endif /* FB_OPERATIONS_blas_H */
