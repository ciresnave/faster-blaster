/**
 * @file judge_op_ids.h
 * @brief Canonical operation ID assignments for the judge metadata table.
 *
 * Op IDs index into fb_op_judge_table[] and fb_benchmark_stats_t arrays.
 * Total: FB_JUDGE_MAX_OPERATIONS (1248) entries.
 *
 * Block assignments:
 *   0   –  47 : BLAS Level 1
 *   48  – 129 : BLAS Level 2
 *   130 – 249 : BLAS Level 3
 *   250 – 749 : LAPACK (computational, driver, eigenvalue, decomposition)
 *   750 – 849 : Sparse
 *   850 – 899 : FFT
 *   900 – 999 : Tensor / DNN extensions
 *  1000 – 1247: Statistics, parallel primitives, chemistry, and other extensions
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_OP_IDS_H
#define FB_JUDGE_OP_IDS_H

/* =========================================================================
 * BLAS Level 1  (0 – 47)
 * ========================================================================= */

/* AXPY: y := alpha*x + y */
#define FB_OP_SAXPY    0
#define FB_OP_DAXPY    1
#define FB_OP_CAXPY    2
#define FB_OP_ZAXPY    3

/* SCAL: x := alpha*x */
#define FB_OP_SSCAL    4
#define FB_OP_DSCAL    5
#define FB_OP_CSCAL    6
#define FB_OP_ZSCAL    7
#define FB_OP_CSSCAL   8   /* complex*real scale */
#define FB_OP_ZDSCAL   9

/* COPY: y := x */
#define FB_OP_SCOPY   10
#define FB_OP_DCOPY   11
#define FB_OP_CCOPY   12
#define FB_OP_ZCOPY   13

/* SWAP: x <-> y */
#define FB_OP_SSWAP   14
#define FB_OP_DSWAP   15
#define FB_OP_CSWAP   16
#define FB_OP_ZSWAP   17

/* DOT: scalar = x^T y */
#define FB_OP_SDOT    18
#define FB_OP_DDOT    19
#define FB_OP_CDOTC   20   /* conjugate dot */
#define FB_OP_CDOTU   21   /* unconjugated dot */
#define FB_OP_ZDOTC   22
#define FB_OP_ZDOTU   23
#define FB_OP_SDSDOT  24   /* mixed precision */
#define FB_OP_DSDOT   25

/* ASUM: scalar = sum |x_i| */
#define FB_OP_SASUM   26
#define FB_OP_DASUM   27
#define FB_OP_SCASUM  28
#define FB_OP_DZASUM  29

/* NRM2: scalar = ||x||_2 */
#define FB_OP_SNRM2   30
#define FB_OP_DNRM2   31
#define FB_OP_SCNRM2  32
#define FB_OP_DZNRM2  33

/* AMAX index */
#define FB_OP_ISAMAX  34
#define FB_OP_IDAMAX  35
#define FB_OP_ICAMAX  36
#define FB_OP_IZAMAX  37

/* ROT / ROTG / ROTM / ROTMG */
#define FB_OP_SROT    38
#define FB_OP_DROT    39
#define FB_OP_CROT    40
#define FB_OP_ZROT    41
#define FB_OP_ZDROT   42
#define FB_OP_SROTG   43
#define FB_OP_DROTG   44
#define FB_OP_SROTM   45
#define FB_OP_DROTM   46
#define FB_OP_SROTMG  47   /* last L1 slot */

/* =========================================================================
 * BLAS Level 2  (48 – 129)
 * ========================================================================= */

#define FB_OP_SGEMV   48
#define FB_OP_DGEMV   49
#define FB_OP_CGEMV   50
#define FB_OP_ZGEMV   51

#define FB_OP_SSYMV   52
#define FB_OP_DSYMV   53
#define FB_OP_CHEMV   54
#define FB_OP_ZHEMV   55

#define FB_OP_STRMV   56
#define FB_OP_DTRMV   57
#define FB_OP_CTRMV   58
#define FB_OP_ZTRMV   59

#define FB_OP_STRSV   60
#define FB_OP_DTRSV   61
#define FB_OP_CTRSV   62
#define FB_OP_ZTRSV   63

#define FB_OP_SGER    64
#define FB_OP_DGER    65
#define FB_OP_CGERU   66
#define FB_OP_CGERC   67
#define FB_OP_ZGERU   68
#define FB_OP_ZGERC   69

#define FB_OP_SSYR    70
#define FB_OP_DSYR    71
#define FB_OP_CHER    72
#define FB_OP_ZHER    73
#define FB_OP_SSYR2   74
#define FB_OP_DSYR2   75
#define FB_OP_CHER2   76
#define FB_OP_ZHER2   77

#define FB_OP_SSPMV   78
#define FB_OP_DSPMV   79
#define FB_OP_CHPMV   80
#define FB_OP_ZHPMV   81
#define FB_OP_SSBMV   82
#define FB_OP_DSBMV   83
#define FB_OP_CHBMV   84
#define FB_OP_ZHBMV   85
#define FB_OP_STBMV   86
#define FB_OP_DTBMV   87
#define FB_OP_CTBMV   88
#define FB_OP_ZTBMV   89
#define FB_OP_STBSV   90
#define FB_OP_DTBSV   91
#define FB_OP_CTBSV   92
#define FB_OP_ZTBSV   93
#define FB_OP_STPMV   94
#define FB_OP_DTPMV   95
#define FB_OP_CTPMV   96
#define FB_OP_ZTPMV   97
#define FB_OP_STPSV   98
#define FB_OP_DTPSV   99
#define FB_OP_CTPSV  100
#define FB_OP_ZTPSV  101
#define FB_OP_SSPR   102
#define FB_OP_DSPR   103
#define FB_OP_CHPR   104
#define FB_OP_ZHPR   105
#define FB_OP_SSPR2  106
#define FB_OP_DSPR2  107
#define FB_OP_CHPR2  108
#define FB_OP_ZHPR2  109
/* 110–129: reserved for additional L2 variants */

/* =========================================================================
 * BLAS Level 3  (130 – 249)
 * ========================================================================= */

#define FB_OP_SGEMM  130
#define FB_OP_DGEMM  131
#define FB_OP_CGEMM  132
#define FB_OP_ZGEMM  133

#define FB_OP_SSYMM  134
#define FB_OP_DSYMM  135
#define FB_OP_CSYMM  136
#define FB_OP_ZSYMM  137
#define FB_OP_CHEMM  138
#define FB_OP_ZHEMM  139

#define FB_OP_SSYRK  140
#define FB_OP_DSYRK  141
#define FB_OP_CSYRK  142
#define FB_OP_ZSYRK  143
#define FB_OP_CHERK  144
#define FB_OP_ZHERK  145
#define FB_OP_SSYR2K 146
#define FB_OP_DSYR2K 147
#define FB_OP_CSYR2K 148
#define FB_OP_ZSYR2K 149
#define FB_OP_CHER2K 150
#define FB_OP_ZHER2K 151

#define FB_OP_STRMM  152
#define FB_OP_DTRMM  153
#define FB_OP_CTRMM  154
#define FB_OP_ZTRMM  155

#define FB_OP_STRSM  156
#define FB_OP_DTRSM  157
#define FB_OP_CTRSM  158
#define FB_OP_ZTRSM  159

/* Batched GEMM  (array-of-pointers) */
#define FB_OP_SGEMM_BATCH  160
#define FB_OP_DGEMM_BATCH  161
#define FB_OP_CGEMM_BATCH  162
#define FB_OP_ZGEMM_BATCH  163

/* Batched GEMM  (strided) */
#define FB_OP_SGEMM_STRIDED  164
#define FB_OP_DGEMM_STRIDED  165
#define FB_OP_CGEMM_STRIDED  166
#define FB_OP_ZGEMM_STRIDED  167

/* 168–249: reserved for extended L3 / mixed-precision GEMM variants */

/* =========================================================================
 * LAPACK  (250 – 749)  — filled as Phase 3/4/5 judge archetypes are implemented
 * ========================================================================= */

#define FB_OP_SGETRF 250
#define FB_OP_DGETRF 251
#define FB_OP_CGETRF 252
#define FB_OP_ZGETRF 253
#define FB_OP_SGETRS 254
#define FB_OP_DGETRS 255
#define FB_OP_SPOTRF 260
#define FB_OP_DPOTRF 261
#define FB_OP_SPOTRS 264
#define FB_OP_DPOTRS 265
#define FB_OP_SGEQRF 270
#define FB_OP_DGEQRF 271
#define FB_OP_SGESV  280
#define FB_OP_DGESV  281
#define FB_OP_SSYEV  290
#define FB_OP_DSYEV  291
#define FB_OP_CHEEV  292
#define FB_OP_ZHEEV  293
#define FB_OP_SGESVD 300
#define FB_OP_DGESVD 301
#define FB_OP_CGESVD 302
#define FB_OP_ZGESVD 303
#define FB_OP_SGEEV  310
#define FB_OP_DGEEV  311
/* 312–749: additional LAPACK routines */

#endif /* FB_JUDGE_OP_IDS_H */
