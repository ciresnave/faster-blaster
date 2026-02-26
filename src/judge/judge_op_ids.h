/**
 * @file judge_op_ids.h
 * @brief Canonical operation ID assignments for the judge metadata table.
 *
 * Op IDs are globally unique integers that index into fb_op_judge_table[]
 * and fb_benchmark_stats_t arrays.  Every operation in the faster-blaster
 * operation superset has exactly one ID here.  Archetype-specific source
 * files consume IDs from the ranges assigned below; they do not define
 * their own ID namespaces.
 *
 * An ID being defined here does NOT imply a judge implementation exists yet.
 * Unimplemented slots return FB_JUDGE_NOT_IMPLEMENTED at runtime.
 *
 * Block assignments (family-first ordering, S→D→C→Z within each family):
 *       0 –    48 : BLAS Level 1
 *      49 –   118 : BLAS Level 2
 *     119 –   156 : BLAS Level 3 (core + batched GEMM)
 *     157 –  1321 : LAPACK — standard s/d/c/z routines
 *    1322 –  1633 : ScaLAPACK — parallel p* distributed routines
 *    1634 –  1807 : Extended BLAS (cblas_batch / cblas_strided / axpby variants)
 *    1808 –  1850 : MKL extensions (mkl_jit, mkl_*omatcopy, mkl_sparse_*)
 *    1851 –  1903 : Deep Neural Network primitives (fb_dnn_*)
 *    1904 –  1942 : FFT (fb_fft_*)
 *    1943 –  2000 : Sparse linear algebra (fb_sparse_*)
 *    2001 –  2117 : Tensor operations + RNG + NCCL/RCCL collectives
 *    2118 –  2141 : Statistics + Machine Learning
 *    2142 –  2176 : Parallel Primitives (fb_prim_*)
 *    2177 –  2184 : Chemistry / Physics extensions
 *    2185 –  2217 : Vector math (fb_v*)
 *    2218 –  2233 : Spline interpolation (fb_spline_*)
 *    2234 –  2265 : GEMM / LPGEMM fused extensions
 *
 * FB_JUDGE_MAX_OPERATIONS: 2400
 *
 * Total concrete operation names covered: 2266
 * (see FASTER-BLASTER-OPERATIONS-LIST-APPENDIX.md)
 *
 * Use the FB_OP__*_BEGIN / _END sentinels (not raw numbers) wherever
 * section boundary checks are needed in C source.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_OP_IDS_H
#define FB_JUDGE_OP_IDS_H

#ifndef FB_JUDGE_MAX_OPERATIONS
#define FB_JUDGE_MAX_OPERATIONS 2400
#endif

/* Section boundary sentinels — use these instead of raw numbers. */
#define FB_OP__BLAS_L1_BEGIN                         0
#define FB_OP__BLAS_L1_END                           49
#define FB_OP__BLAS_L2_BEGIN                         49
#define FB_OP__BLAS_L2_END                           119
#define FB_OP__BLAS_L3_BEGIN                         119
#define FB_OP__BLAS_L3_END                           157
#define FB_OP__LAPACK_BEGIN                          157
#define FB_OP__LAPACK_END                            1322
#define FB_OP__SCALAPACK_BEGIN                       1322
#define FB_OP__SCALAPACK_END                         1634

/* =========================================================================
 * BLAS Level 1  (0 – 48)
 * ========================================================================= */

#define FB_OP_SASUM                                  0
#define FB_OP_DASUM                                  1

#define FB_OP_SCASUM                                 2

#define FB_OP_DZASUM                                 3

#define FB_OP_SAXPY                                  4
#define FB_OP_DAXPY                                  5
#define FB_OP_CAXPY                                  6
#define FB_OP_ZAXPY                                  7

#define FB_OP_SCOPY                                  8
#define FB_OP_DCOPY                                  9
#define FB_OP_CCOPY                                  10
#define FB_OP_ZCOPY                                  11

#define FB_OP_SDOT                                   12
#define FB_OP_DDOT                                   13

#define FB_OP_CDOTC                                  14
#define FB_OP_ZDOTC                                  15

#define FB_OP_CDOTU                                  16
#define FB_OP_ZDOTU                                  17

#define FB_OP_ZDROT                                  18

#define FB_OP_SDSDOT                                 19

#define FB_OP_DSDOT                                  20

#define FB_OP_ISAMAX                                 21

#define FB_OP_IDAMAX                                 22

#define FB_OP_ICAMAX                                 23

#define FB_OP_IZAMAX                                 24

#define FB_OP_SNRM2                                  25
#define FB_OP_DNRM2                                  26

#define FB_OP_SCNRM2                                 27

#define FB_OP_DZNRM2                                 28

#define FB_OP_SROT                                   29
#define FB_OP_DROT                                   30
#define FB_OP_CROT                                   31
#define FB_OP_ZROT                                   32

#define FB_OP_SROTG                                  33
#define FB_OP_DROTG                                  34

#define FB_OP_SROTM                                  35
#define FB_OP_DROTM                                  36

#define FB_OP_SROTMG                                 37
#define FB_OP_DROTMG                                 38

#define FB_OP_SSCAL                                  39
#define FB_OP_DSCAL                                  40
#define FB_OP_CSCAL                                  41

#define FB_OP_CSSCAL                                 42

#define FB_OP_ZSCAL                                  43

#define FB_OP_ZDSCAL                                 44

#define FB_OP_SSWAP                                  45
#define FB_OP_DSWAP                                  46
#define FB_OP_CSWAP                                  47
#define FB_OP_ZSWAP                                  48

/* =========================================================================
 * BLAS Level 2  (49 – 118)
 * ========================================================================= */

#define FB_OP_SGBMV                                  49
#define FB_OP_DGBMV                                  50
#define FB_OP_CGBMV                                  51
#define FB_OP_ZGBMV                                  52

#define FB_OP_SGBMVX                                 53
#define FB_OP_DGBMVX                                 54
#define FB_OP_CGBMVX                                 55
#define FB_OP_ZGBMVX                                 56

#define FB_OP_SGEMV                                  57
#define FB_OP_DGEMV                                  58
#define FB_OP_CGEMV                                  59
#define FB_OP_ZGEMV                                  60

#define FB_OP_SGER                                   61
#define FB_OP_DGER                                   62

#define FB_OP_CGERC                                  63
#define FB_OP_ZGERC                                  64

#define FB_OP_CGERU                                  65
#define FB_OP_ZGERU                                  66

#define FB_OP_CHBMV                                  67
#define FB_OP_ZHBMV                                  68

#define FB_OP_CHEMV                                  69
#define FB_OP_ZHEMV                                  70

#define FB_OP_CHER                                   71
#define FB_OP_ZHER                                   72

#define FB_OP_CHER2                                  73
#define FB_OP_ZHER2                                  74

#define FB_OP_CHPMV                                  75
#define FB_OP_ZHPMV                                  76

#define FB_OP_CHPR                                   77
#define FB_OP_ZHPR                                   78

#define FB_OP_CHPR2                                  79
#define FB_OP_ZHPR2                                  80

#define FB_OP_SSBMV                                  81
#define FB_OP_DSBMV                                  82

#define FB_OP_SSPMV                                  83
#define FB_OP_DSPMV                                  84

#define FB_OP_SSPR                                   85
#define FB_OP_DSPR                                   86

#define FB_OP_SSPR2                                  87
#define FB_OP_DSPR2                                  88

#define FB_OP_SSYMV                                  89
#define FB_OP_DSYMV                                  90

#define FB_OP_SSYR                                   91
#define FB_OP_DSYR                                   92

#define FB_OP_SSYR2                                  93
#define FB_OP_DSYR2                                  94

#define FB_OP_STBMV                                  95
#define FB_OP_DTBMV                                  96
#define FB_OP_CTBMV                                  97
#define FB_OP_ZTBMV                                  98

#define FB_OP_STBSV                                  99
#define FB_OP_DTBSV                                  100
#define FB_OP_CTBSV                                  101
#define FB_OP_ZTBSV                                  102

#define FB_OP_STPMV                                  103
#define FB_OP_DTPMV                                  104
#define FB_OP_CTPMV                                  105
#define FB_OP_ZTPMV                                  106

#define FB_OP_STPSV                                  107
#define FB_OP_DTPSV                                  108
#define FB_OP_CTPSV                                  109
#define FB_OP_ZTPSV                                  110

#define FB_OP_STRMV                                  111
#define FB_OP_DTRMV                                  112
#define FB_OP_CTRMV                                  113
#define FB_OP_ZTRMV                                  114

#define FB_OP_STRSV                                  115
#define FB_OP_DTRSV                                  116
#define FB_OP_CTRSV                                  117
#define FB_OP_ZTRSV                                  118

/* =========================================================================
 * BLAS Level 3 (core + batched GEMM)  (119 – 156)
 * ========================================================================= */

#define FB_OP_SGEMM                                  119
#define FB_OP_DGEMM                                  120
#define FB_OP_CGEMM                                  121
#define FB_OP_ZGEMM                                  122

#define FB_OP_SGEMM_BATCH                            123
#define FB_OP_DGEMM_BATCH                            124
#define FB_OP_CGEMM_BATCH                            125
#define FB_OP_ZGEMM_BATCH                            126

#define FB_OP_SGEMM_STRIDED                          127
#define FB_OP_DGEMM_STRIDED                          128
#define FB_OP_CGEMM_STRIDED                          129
#define FB_OP_ZGEMM_STRIDED                          130

#define FB_OP_CHEMM                                  131
#define FB_OP_ZHEMM                                  132

#define FB_OP_CHER2K                                 133
#define FB_OP_ZHER2K                                 134

#define FB_OP_CHERK                                  135
#define FB_OP_ZHERK                                  136

#define FB_OP_SSYMM                                  137
#define FB_OP_DSYMM                                  138
#define FB_OP_CSYMM                                  139
#define FB_OP_ZSYMM                                  140

#define FB_OP_SSYR2K                                 141
#define FB_OP_DSYR2K                                 142
#define FB_OP_CSYR2K                                 143
#define FB_OP_ZSYR2K                                 144

#define FB_OP_SSYRK                                  145
#define FB_OP_DSYRK                                  146
#define FB_OP_CSYRK                                  147
#define FB_OP_ZSYRK                                  148

#define FB_OP_STRMM                                  149
#define FB_OP_DTRMM                                  150
#define FB_OP_CTRMM                                  151
#define FB_OP_ZTRMM                                  152

#define FB_OP_STRSM                                  153
#define FB_OP_DTRSM                                  154
#define FB_OP_CTRSM                                  155
#define FB_OP_ZTRSM                                  156

/* =========================================================================
 * LAPACK — standard s/d/c/z routines  (157 – 1321)
 * ========================================================================= */

#define FB_OP_SBDSDC                                 157
#define FB_OP_DBDSDC                                 158
#define FB_OP_CBDSDC                                 159
#define FB_OP_ZBDSDC                                 160

#define FB_OP_SBDSQR                                 161
#define FB_OP_DBDSQR                                 162
#define FB_OP_CBDSQR                                 163
#define FB_OP_ZBDSQR                                 164

#define FB_OP_SBMV                                   165
#define FB_OP_DBMV                                   166

#define FB_OP_SCSUM1                                 167

#define FB_OP_ZDRSCL                                 168

#define FB_OP_SGBCON                                 169
#define FB_OP_DGBCON                                 170
#define FB_OP_CGBCON                                 171
#define FB_OP_ZGBCON                                 172

#define FB_OP_SGBEQU                                 173
#define FB_OP_DGBEQU                                 174
#define FB_OP_CGBEQU                                 175
#define FB_OP_ZGBEQU                                 176

#define FB_OP_SGBRFS                                 177
#define FB_OP_DGBRFS                                 178
#define FB_OP_CGBRFS                                 179
#define FB_OP_ZGBRFS                                 180

#define FB_OP_SGBSV                                  181
#define FB_OP_DGBSV                                  182
#define FB_OP_CGBSV                                  183
#define FB_OP_ZGBSV                                  184

#define FB_OP_SGBSVX                                 185
#define FB_OP_DGBSVX                                 186
#define FB_OP_CGBSVX                                 187
#define FB_OP_ZGBSVX                                 188

#define FB_OP_SGBTRF                                 189
#define FB_OP_DGBTRF                                 190
#define FB_OP_CGBTRF                                 191
#define FB_OP_ZGBTRF                                 192

#define FB_OP_SGBTRS                                 193
#define FB_OP_DGBTRS                                 194
#define FB_OP_CGBTRS                                 195
#define FB_OP_ZGBTRS                                 196

#define FB_OP_SGEBAK                                 197
#define FB_OP_DGEBAK                                 198
#define FB_OP_CGEBAK                                 199
#define FB_OP_ZGEBAK                                 200

#define FB_OP_SGEBAL                                 201
#define FB_OP_DGEBAL                                 202
#define FB_OP_CGEBAL                                 203
#define FB_OP_ZGEBAL                                 204

#define FB_OP_SGEBD2                                 205
#define FB_OP_DGEBD2                                 206
#define FB_OP_CGEBD2                                 207
#define FB_OP_ZGEBD2                                 208

#define FB_OP_SGEBRD                                 209
#define FB_OP_DGEBRD                                 210
#define FB_OP_CGEBRD                                 211
#define FB_OP_ZGEBRD                                 212

#define FB_OP_SGECON                                 213
#define FB_OP_DGECON                                 214
#define FB_OP_CGECON                                 215
#define FB_OP_ZGECON                                 216

#define FB_OP_SGEEQU                                 217
#define FB_OP_DGEEQU                                 218
#define FB_OP_CGEEQU                                 219
#define FB_OP_ZGEEQU                                 220

#define FB_OP_SGEES                                  221
#define FB_OP_DGEES                                  222
#define FB_OP_CGEES                                  223
#define FB_OP_ZGEES                                  224

#define FB_OP_SGEESX                                 225
#define FB_OP_DGEESX                                 226
#define FB_OP_CGEESX                                 227
#define FB_OP_ZGEESX                                 228

#define FB_OP_SGEEV                                  229
#define FB_OP_DGEEV                                  230
#define FB_OP_CGEEV                                  231
#define FB_OP_ZGEEV                                  232

#define FB_OP_SGEEVX                                 233
#define FB_OP_DGEEVX                                 234
#define FB_OP_CGEEVX                                 235
#define FB_OP_ZGEEVX                                 236

#define FB_OP_SGEHD2                                 237
#define FB_OP_DGEHD2                                 238
#define FB_OP_CGEHD2                                 239
#define FB_OP_ZGEHD2                                 240

#define FB_OP_SGEHRD                                 241
#define FB_OP_DGEHRD                                 242
#define FB_OP_CGEHRD                                 243
#define FB_OP_ZGEHRD                                 244

#define FB_OP_SGELQ2                                 245
#define FB_OP_DGELQ2                                 246
#define FB_OP_CGELQ2                                 247
#define FB_OP_ZGELQ2                                 248

#define FB_OP_SGELQF                                 249
#define FB_OP_DGELQF                                 250
#define FB_OP_CGELQF                                 251
#define FB_OP_ZGELQF                                 252

#define FB_OP_SGELS                                  253
#define FB_OP_DGELS                                  254
#define FB_OP_CGELS                                  255
#define FB_OP_ZGELS                                  256

#define FB_OP_SGELSD                                 257
#define FB_OP_DGELSD                                 258
#define FB_OP_CGELSD                                 259
#define FB_OP_ZGELSD                                 260

#define FB_OP_SGELSS                                 261
#define FB_OP_DGELSS                                 262
#define FB_OP_CGELSS                                 263
#define FB_OP_ZGELSS                                 264

#define FB_OP_SGELSY                                 265
#define FB_OP_DGELSY                                 266
#define FB_OP_CGELSY                                 267
#define FB_OP_ZGELSY                                 268

#define FB_OP_SGEMMT                                 269
#define FB_OP_DGEMMT                                 270
#define FB_OP_CGEMMT                                 271
#define FB_OP_ZGEMMT                                 272

#define FB_OP_SGEMM_COMPUTE                          273
#define FB_OP_DGEMM_COMPUTE                          274
#define FB_OP_CGEMM_COMPUTE                          275
#define FB_OP_ZGEMM_COMPUTE                          276

#define FB_OP_SGEMM_PACK                             277
#define FB_OP_DGEMM_PACK                             278
#define FB_OP_CGEMM_PACK                             279
#define FB_OP_ZGEMM_PACK                             280

#define FB_OP_SGEMM_PACK_GET_SIZE                    281
#define FB_OP_DGEMM_PACK_GET_SIZE                    282
#define FB_OP_CGEMM_PACK_GET_SIZE                    283
#define FB_OP_ZGEMM_PACK_GET_SIZE                    284

#define FB_OP_SGEMM_PTR                              285
#define FB_OP_DGEMM_PTR                              286
#define FB_OP_CGEMM_PTR                              287
#define FB_OP_ZGEMM_PTR                              288

#define FB_OP_SGEQL2                                 289
#define FB_OP_DGEQL2                                 290
#define FB_OP_CGEQL2                                 291
#define FB_OP_ZGEQL2                                 292

#define FB_OP_SGEQLF                                 293
#define FB_OP_DGEQLF                                 294
#define FB_OP_CGEQLF                                 295
#define FB_OP_ZGEQLF                                 296

#define FB_OP_SGEQP3                                 297
#define FB_OP_DGEQP3                                 298
#define FB_OP_CGEQP3                                 299
#define FB_OP_ZGEQP3                                 300

#define FB_OP_SGEQPF                                 301
#define FB_OP_DGEQPF                                 302
#define FB_OP_CGEQPF                                 303
#define FB_OP_ZGEQPF                                 304

#define FB_OP_SGEQRF                                 305
#define FB_OP_DGEQRF                                 306
#define FB_OP_CGEQRF                                 307
#define FB_OP_ZGEQRF                                 308

#define FB_OP_SGERFS                                 309
#define FB_OP_DGERFS                                 310
#define FB_OP_CGERFS                                 311
#define FB_OP_ZGERFS                                 312

#define FB_OP_SGERQ2                                 313
#define FB_OP_DGERQ2                                 314
#define FB_OP_CGERQ2                                 315
#define FB_OP_ZGERQ2                                 316

#define FB_OP_SGERQF                                 317
#define FB_OP_DGERQF                                 318
#define FB_OP_CGERQF                                 319
#define FB_OP_ZGERQF                                 320

#define FB_OP_SGESDD                                 321
#define FB_OP_DGESDD                                 322
#define FB_OP_CGESDD                                 323
#define FB_OP_ZGESDD                                 324

#define FB_OP_SGESV                                  325
#define FB_OP_DGESV                                  326
#define FB_OP_CGESV                                  327
#define FB_OP_ZGESV                                  328

#define FB_OP_SGESVD                                 329
#define FB_OP_DGESVD                                 330
#define FB_OP_CGESVD                                 331
#define FB_OP_ZGESVD                                 332

#define FB_OP_SGESVX                                 333
#define FB_OP_DGESVX                                 334
#define FB_OP_CGESVX                                 335
#define FB_OP_ZGESVX                                 336

#define FB_OP_SGESVXX                                337
#define FB_OP_DGESVXX                                338
#define FB_OP_CGESVXX                                339
#define FB_OP_ZGESVXX                                340

#define FB_OP_SGETRF                                 341
#define FB_OP_DGETRF                                 342
#define FB_OP_CGETRF                                 343
#define FB_OP_ZGETRF                                 344

#define FB_OP_SGETRI                                 345
#define FB_OP_DGETRI                                 346
#define FB_OP_CGETRI                                 347
#define FB_OP_ZGETRI                                 348

#define FB_OP_SGETRS                                 349
#define FB_OP_DGETRS                                 350
#define FB_OP_CGETRS                                 351
#define FB_OP_ZGETRS                                 352

#define FB_OP_SGGES                                  353
#define FB_OP_DGGES                                  354
#define FB_OP_CGGES                                  355
#define FB_OP_ZGGES                                  356

#define FB_OP_SGGESX                                 357
#define FB_OP_DGGESX                                 358
#define FB_OP_CGGESX                                 359
#define FB_OP_ZGGESX                                 360

#define FB_OP_SGGEV                                  361
#define FB_OP_DGGEV                                  362
#define FB_OP_CGGEV                                  363
#define FB_OP_ZGGEV                                  364

#define FB_OP_SGGEVX                                 365
#define FB_OP_DGGEVX                                 366
#define FB_OP_CGGEVX                                 367
#define FB_OP_ZGGEVX                                 368

#define FB_OP_SGGGLM                                 369
#define FB_OP_DGGGLM                                 370
#define FB_OP_CGGGLM                                 371
#define FB_OP_ZGGGLM                                 372

#define FB_OP_SGGHRD                                 373
#define FB_OP_DGGHRD                                 374
#define FB_OP_CGGHRD                                 375
#define FB_OP_ZGGHRD                                 376

#define FB_OP_SGGLSE                                 377
#define FB_OP_DGGLSE                                 378
#define FB_OP_CGGLSE                                 379
#define FB_OP_ZGGLSE                                 380

#define FB_OP_SGGSVD                                 381
#define FB_OP_DGGSVD                                 382
#define FB_OP_CGGSVD                                 383
#define FB_OP_ZGGSVD                                 384

#define FB_OP_SGTCON                                 385
#define FB_OP_DGTCON                                 386
#define FB_OP_CGTCON                                 387
#define FB_OP_ZGTCON                                 388

#define FB_OP_SGTRFS                                 389
#define FB_OP_DGTRFS                                 390
#define FB_OP_CGTRFS                                 391
#define FB_OP_ZGTRFS                                 392

#define FB_OP_SGTSV                                  393
#define FB_OP_DGTSV                                  394
#define FB_OP_CGTSV                                  395
#define FB_OP_ZGTSV                                  396

#define FB_OP_SGTSVX                                 397
#define FB_OP_DGTSVX                                 398
#define FB_OP_CGTSVX                                 399
#define FB_OP_ZGTSVX                                 400

#define FB_OP_SGTTRF                                 401
#define FB_OP_DGTTRF                                 402
#define FB_OP_CGTTRF                                 403
#define FB_OP_ZGTTRF                                 404

#define FB_OP_SGTTRS                                 405
#define FB_OP_DGTTRS                                 406
#define FB_OP_CGTTRS                                 407
#define FB_OP_ZGTTRS                                 408

#define FB_OP_CHBEV                                  409
#define FB_OP_ZHBEV                                  410

#define FB_OP_CHBEVD                                 411
#define FB_OP_ZHBEVD                                 412

#define FB_OP_CHBEVX                                 413
#define FB_OP_ZHBEVX                                 414

#define FB_OP_CHBGV                                  415
#define FB_OP_ZHBGV                                  416

#define FB_OP_CHBGVD                                 417
#define FB_OP_ZHBGVD                                 418

#define FB_OP_CHBGVX                                 419
#define FB_OP_ZHBGVX                                 420

#define FB_OP_SHBMV                                  421
#define FB_OP_DHBMV                                  422

#define FB_OP_CHBTRF                                 423
#define FB_OP_ZHBTRF                                 424

#define FB_OP_CHECON                                 425
#define FB_OP_ZHECON                                 426

#define FB_OP_CHEEV                                  427
#define FB_OP_ZHEEV                                  428

#define FB_OP_CHEEVD                                 429
#define FB_OP_ZHEEVD                                 430

#define FB_OP_CHEEVR                                 431
#define FB_OP_ZHEEVR                                 432

#define FB_OP_CHEEVX                                 433
#define FB_OP_ZHEEVX                                 434

#define FB_OP_CHEGST                                 435
#define FB_OP_ZHEGST                                 436

#define FB_OP_CHEGV                                  437
#define FB_OP_ZHEGV                                  438

#define FB_OP_CHEGVD                                 439
#define FB_OP_ZHEGVD                                 440

#define FB_OP_CHEGVX                                 441
#define FB_OP_ZHEGVX                                 442

#define FB_OP_SHER                                   443
#define FB_OP_DHER                                   444

#define FB_OP_SHER2                                  445
#define FB_OP_DHER2                                  446

#define FB_OP_CHERFS                                 447
#define FB_OP_ZHERFS                                 448

#define FB_OP_CHESV                                  449
#define FB_OP_ZHESV                                  450

#define FB_OP_CHESVX                                 451
#define FB_OP_ZHESVX                                 452

#define FB_OP_CHESVXX                                453
#define FB_OP_ZHESVXX                                454

#define FB_OP_SHETD2                                 455
#define FB_OP_DHETD2                                 456

#define FB_OP_CHETRD                                 457
#define FB_OP_ZHETRD                                 458

#define FB_OP_CHETRF                                 459
#define FB_OP_ZHETRF                                 460

#define FB_OP_CHETRF_ROOK                            461
#define FB_OP_ZHETRF_ROOK                            462

#define FB_OP_CHETRI                                 463
#define FB_OP_ZHETRI                                 464

#define FB_OP_CHETRS                                 465
#define FB_OP_ZHETRS                                 466

#define FB_OP_SHGEQZ                                 467
#define FB_OP_DHGEQZ                                 468
#define FB_OP_CHGEQZ                                 469
#define FB_OP_ZHGEQZ                                 470

#define FB_OP_CHPCON                                 471
#define FB_OP_ZHPCON                                 472

#define FB_OP_CHPEV                                  473
#define FB_OP_ZHPEV                                  474

#define FB_OP_CHPEVD                                 475
#define FB_OP_ZHPEVD                                 476

#define FB_OP_CHPEVX                                 477
#define FB_OP_ZHPEVX                                 478

#define FB_OP_CHPGV                                  479
#define FB_OP_ZHPGV                                  480

#define FB_OP_CHPGVD                                 481
#define FB_OP_ZHPGVD                                 482

#define FB_OP_CHPGVX                                 483
#define FB_OP_ZHPGVX                                 484

#define FB_OP_SHPMV                                  485
#define FB_OP_DHPMV                                  486

#define FB_OP_CHPRFS                                 487
#define FB_OP_ZHPRFS                                 488

#define FB_OP_CHPSV                                  489
#define FB_OP_ZHPSV                                  490

#define FB_OP_CHPTRF                                 491
#define FB_OP_ZHPTRF                                 492

#define FB_OP_CHPTRI                                 493
#define FB_OP_ZHPTRI                                 494

#define FB_OP_CHPTRS                                 495
#define FB_OP_ZHPTRS                                 496

#define FB_OP_SHSEQR                                 497
#define FB_OP_DHSEQR                                 498
#define FB_OP_CHSEQR                                 499
#define FB_OP_ZHSEQR                                 500

#define FB_OP_SIMATCOPY                              501
#define FB_OP_DIMATCOPY                              502
#define FB_OP_CIMATCOPY                              503
#define FB_OP_ZIMATCOPY                              504

#define FB_OP_SLABAD                                 505
#define FB_OP_DLABAD                                 506

#define FB_OP_SLABRD                                 507
#define FB_OP_DLABRD                                 508
#define FB_OP_CLABRD                                 509
#define FB_OP_ZLABRD                                 510

#define FB_OP_CLACGV                                 511
#define FB_OP_ZLACGV                                 512

#define FB_OP_SLACON                                 513
#define FB_OP_DLACON                                 514
#define FB_OP_CLACON                                 515
#define FB_OP_ZLACON                                 516

#define FB_OP_SLACPY                                 517
#define FB_OP_DLACPY                                 518
#define FB_OP_CLACPY                                 519
#define FB_OP_ZLACPY                                 520

#define FB_OP_CLACRM                                 521
#define FB_OP_ZLACRM                                 522

#define FB_OP_CLACRT                                 523
#define FB_OP_ZLACRT                                 524

#define FB_OP_SLADIV                                 525
#define FB_OP_DLADIV                                 526

#define FB_OP_SLAEBZ                                 527
#define FB_OP_DLAEBZ                                 528

#define FB_OP_SLAED0                                 529
#define FB_OP_DLAED0                                 530

#define FB_OP_SLAED1                                 531
#define FB_OP_DLAED1                                 532

#define FB_OP_SLAED2                                 533
#define FB_OP_DLAED2                                 534

#define FB_OP_SLAED3                                 535
#define FB_OP_DLAED3                                 536

#define FB_OP_SLAED4                                 537
#define FB_OP_DLAED4                                 538

#define FB_OP_SLAED5                                 539
#define FB_OP_DLAED5                                 540

#define FB_OP_SLAED6                                 541
#define FB_OP_DLAED6                                 542

#define FB_OP_SLAED7                                 543
#define FB_OP_DLAED7                                 544

#define FB_OP_SLAED8                                 545
#define FB_OP_DLAED8                                 546

#define FB_OP_SLAED9                                 547
#define FB_OP_DLAED9                                 548

#define FB_OP_SLAEDA                                 549
#define FB_OP_DLAEDA                                 550

#define FB_OP_SLAEIN                                 551
#define FB_OP_DLAEIN                                 552
#define FB_OP_CLAEIN                                 553
#define FB_OP_ZLAEIN                                 554

#define FB_OP_CLAESY                                 555
#define FB_OP_ZLAESY                                 556

#define FB_OP_SLAEV2                                 557
#define FB_OP_DLAEV2                                 558

#define FB_OP_SLAEXC                                 559
#define FB_OP_DLAEXC                                 560
#define FB_OP_CLAEXC                                 561
#define FB_OP_ZLAEXC                                 562

#define FB_OP_SLAGS2                                 563
#define FB_OP_DLAGS2                                 564
#define FB_OP_CLAGS2                                 565
#define FB_OP_ZLAGS2                                 566

#define FB_OP_SLAGTF                                 567
#define FB_OP_DLAGTF                                 568
#define FB_OP_CLAGTF                                 569
#define FB_OP_ZLAGTF                                 570

#define FB_OP_SLAGTM                                 571
#define FB_OP_DLAGTM                                 572
#define FB_OP_CLAGTM                                 573
#define FB_OP_ZLAGTM                                 574

#define FB_OP_SLAGTS                                 575
#define FB_OP_DLAGTS                                 576
#define FB_OP_CLAGTS                                 577
#define FB_OP_ZLAGTS                                 578

#define FB_OP_SLAGV2                                 579
#define FB_OP_DLAGV2                                 580
#define FB_OP_CLAGV2                                 581
#define FB_OP_ZLAGV2                                 582

#define FB_OP_SLAHQR                                 583
#define FB_OP_DLAHQR                                 584
#define FB_OP_CLAHQR                                 585
#define FB_OP_ZLAHQR                                 586

#define FB_OP_SLAHRD                                 587
#define FB_OP_DLAHRD                                 588
#define FB_OP_CLAHRD                                 589
#define FB_OP_ZLAHRD                                 590

#define FB_OP_SLAIC1                                 591
#define FB_OP_DLAIC1                                 592
#define FB_OP_CLAIC1                                 593
#define FB_OP_ZLAIC1                                 594

#define FB_OP_SLALN2                                 595
#define FB_OP_DLALN2                                 596

#define FB_OP_SLAMCH                                 597
#define FB_OP_DLAMCH                                 598

#define FB_OP_SLANGB                                 599
#define FB_OP_DLANGB                                 600
#define FB_OP_CLANGB                                 601
#define FB_OP_ZLANGB                                 602

#define FB_OP_SLANGE                                 603
#define FB_OP_DLANGE                                 604
#define FB_OP_CLANGE                                 605
#define FB_OP_ZLANGE                                 606

#define FB_OP_SLANGT                                 607
#define FB_OP_DLANGT                                 608
#define FB_OP_CLANGT                                 609
#define FB_OP_ZLANGT                                 610

#define FB_OP_CLANHB                                 611
#define FB_OP_ZLANHB                                 612

#define FB_OP_CLANHE                                 613
#define FB_OP_ZLANHE                                 614

#define FB_OP_CLANHP                                 615
#define FB_OP_ZLANHP                                 616

#define FB_OP_SLANHS                                 617
#define FB_OP_DLANHS                                 618
#define FB_OP_CLANHS                                 619
#define FB_OP_ZLANHS                                 620

#define FB_OP_SLANSB                                 621
#define FB_OP_DLANSB                                 622
#define FB_OP_CLANSB                                 623
#define FB_OP_ZLANSB                                 624

#define FB_OP_SLANSP                                 625
#define FB_OP_DLANSP                                 626
#define FB_OP_CLANSP                                 627
#define FB_OP_ZLANSP                                 628

#define FB_OP_SLANST                                 629
#define FB_OP_DLANST                                 630

#define FB_OP_SLANSY                                 631
#define FB_OP_DLANSY                                 632
#define FB_OP_CLANSY                                 633
#define FB_OP_ZLANSY                                 634

#define FB_OP_SLANTB                                 635
#define FB_OP_DLANTB                                 636
#define FB_OP_CLANTB                                 637
#define FB_OP_ZLANTB                                 638

#define FB_OP_SLANTP                                 639
#define FB_OP_DLANTP                                 640
#define FB_OP_CLANTP                                 641
#define FB_OP_ZLANTP                                 642

#define FB_OP_SLANTR                                 643
#define FB_OP_DLANTR                                 644
#define FB_OP_CLANTR                                 645
#define FB_OP_ZLANTR                                 646

#define FB_OP_SLAPMT                                 647
#define FB_OP_DLAPMT                                 648
#define FB_OP_CLAPMT                                 649
#define FB_OP_ZLAPMT                                 650

#define FB_OP_SLAPY2                                 651
#define FB_OP_DLAPY2                                 652

#define FB_OP_SLAPY3                                 653
#define FB_OP_DLAPY3                                 654

#define FB_OP_SLAQGB                                 655
#define FB_OP_DLAQGB                                 656
#define FB_OP_CLAQGB                                 657
#define FB_OP_ZLAQGB                                 658

#define FB_OP_SLAQGE                                 659
#define FB_OP_DLAQGE                                 660
#define FB_OP_CLAQGE                                 661
#define FB_OP_ZLAQGE                                 662

#define FB_OP_CLAQHE                                 663
#define FB_OP_ZLAQHE                                 664

#define FB_OP_CLAQHP                                 665
#define FB_OP_ZLAQHP                                 666

#define FB_OP_SLAQP2                                 667
#define FB_OP_DLAQP2                                 668
#define FB_OP_CLAQP2                                 669
#define FB_OP_ZLAQP2                                 670

#define FB_OP_SLAQPS                                 671
#define FB_OP_DLAQPS                                 672
#define FB_OP_CLAQPS                                 673
#define FB_OP_ZLAQPS                                 674

#define FB_OP_SLAQR0                                 675
#define FB_OP_DLAQR0                                 676

#define FB_OP_SLAQR1                                 677
#define FB_OP_DLAQR1                                 678

#define FB_OP_SLAQR2                                 679
#define FB_OP_DLAQR2                                 680

#define FB_OP_SLAQSB                                 681
#define FB_OP_DLAQSB                                 682
#define FB_OP_CLAQSB                                 683
#define FB_OP_ZLAQSB                                 684

#define FB_OP_SLAQSP                                 685
#define FB_OP_DLAQSP                                 686
#define FB_OP_CLAQSP                                 687
#define FB_OP_ZLAQSP                                 688

#define FB_OP_SLAQSY                                 689
#define FB_OP_DLAQSY                                 690
#define FB_OP_CLAQSY                                 691
#define FB_OP_ZLAQSY                                 692

#define FB_OP_SLAR1V                                 693
#define FB_OP_DLAR1V                                 694

#define FB_OP_SLAR2V                                 695
#define FB_OP_DLAR2V                                 696

#define FB_OP_SLARF                                  697
#define FB_OP_DLARF                                  698
#define FB_OP_CLARF                                  699
#define FB_OP_ZLARF                                  700

#define FB_OP_SLARFB                                 701
#define FB_OP_DLARFB                                 702
#define FB_OP_CLARFB                                 703
#define FB_OP_ZLARFB                                 704

#define FB_OP_SLARFG                                 705
#define FB_OP_DLARFG                                 706
#define FB_OP_CLARFG                                 707
#define FB_OP_ZLARFG                                 708

#define FB_OP_SLARFT                                 709
#define FB_OP_DLARFT                                 710
#define FB_OP_CLARFT                                 711
#define FB_OP_ZLARFT                                 712

#define FB_OP_SLARFX                                 713
#define FB_OP_DLARFX                                 714
#define FB_OP_CLARFX                                 715
#define FB_OP_ZLARFX                                 716

#define FB_OP_SLARGE                                 717
#define FB_OP_DLARGE                                 718

#define FB_OP_CLARGV                                 719
#define FB_OP_ZLARGV                                 720

#define FB_OP_SLARNV                                 721
#define FB_OP_DLARNV                                 722
#define FB_OP_CLARNV                                 723
#define FB_OP_ZLARNV                                 724

#define FB_OP_SLARRA                                 725
#define FB_OP_DLARRA                                 726

#define FB_OP_SLARRB                                 727
#define FB_OP_DLARRB                                 728

#define FB_OP_SLARRC                                 729
#define FB_OP_DLARRC                                 730

#define FB_OP_SLARRD                                 731
#define FB_OP_DLARRD                                 732

#define FB_OP_SLARRE                                 733
#define FB_OP_DLARRE                                 734

#define FB_OP_SLARRF                                 735
#define FB_OP_DLARRF                                 736

#define FB_OP_SLARRJ                                 737
#define FB_OP_DLARRJ                                 738

#define FB_OP_SLARRK                                 739
#define FB_OP_DLARRK                                 740

#define FB_OP_SLARRR                                 741
#define FB_OP_DLARRR                                 742

#define FB_OP_SLARRV                                 743
#define FB_OP_DLARRV                                 744

#define FB_OP_SLARTG                                 745
#define FB_OP_DLARTG                                 746
#define FB_OP_CLARTG                                 747
#define FB_OP_ZLARTG                                 748

#define FB_OP_SLARTV                                 749
#define FB_OP_DLARTV                                 750
#define FB_OP_CLARTV                                 751
#define FB_OP_ZLARTV                                 752

#define FB_OP_SLARZ                                  753
#define FB_OP_DLARZ                                  754
#define FB_OP_CLARZ                                  755
#define FB_OP_ZLARZ                                  756

#define FB_OP_SLARZB                                 757
#define FB_OP_DLARZB                                 758
#define FB_OP_CLARZB                                 759
#define FB_OP_ZLARZB                                 760

#define FB_OP_SLARZT                                 761
#define FB_OP_DLARZT                                 762
#define FB_OP_CLARZT                                 763
#define FB_OP_ZLARZT                                 764

#define FB_OP_SLAS2                                  765
#define FB_OP_DLAS2                                  766

#define FB_OP_SLASCL                                 767
#define FB_OP_DLASCL                                 768
#define FB_OP_CLASCL                                 769
#define FB_OP_ZLASCL                                 770

#define FB_OP_SLASD0                                 771
#define FB_OP_DLASD0                                 772

#define FB_OP_SLASD1                                 773
#define FB_OP_DLASD1                                 774

#define FB_OP_SLASD2                                 775
#define FB_OP_DLASD2                                 776

#define FB_OP_SLASD3                                 777
#define FB_OP_DLASD3                                 778

#define FB_OP_SLASD4                                 779
#define FB_OP_DLASD4                                 780

#define FB_OP_SLASD5                                 781
#define FB_OP_DLASD5                                 782

#define FB_OP_SLASD6                                 783
#define FB_OP_DLASD6                                 784

#define FB_OP_SLASD7                                 785
#define FB_OP_DLASD7                                 786

#define FB_OP_SLASD8                                 787
#define FB_OP_DLASD8                                 788

#define FB_OP_SLASDA                                 789
#define FB_OP_DLASDA                                 790

#define FB_OP_SLASDQ                                 791
#define FB_OP_DLASDQ                                 792

#define FB_OP_SLASDT                                 793
#define FB_OP_DLASDT                                 794

#define FB_OP_SLASET                                 795
#define FB_OP_DLASET                                 796
#define FB_OP_CLASET                                 797
#define FB_OP_ZLASET                                 798

#define FB_OP_SLASQ1                                 799
#define FB_OP_DLASQ1                                 800

#define FB_OP_SLASQ2                                 801
#define FB_OP_DLASQ2                                 802

#define FB_OP_SLASQ3                                 803
#define FB_OP_DLASQ3                                 804

#define FB_OP_SLASQ4                                 805
#define FB_OP_DLASQ4                                 806

#define FB_OP_SLASQ5                                 807
#define FB_OP_DLASQ5                                 808

#define FB_OP_SLASQ6                                 809
#define FB_OP_DLASQ6                                 810

#define FB_OP_SLASR                                  811
#define FB_OP_DLASR                                  812

#define FB_OP_SLASRT                                 813
#define FB_OP_DLASRT                                 814

#define FB_OP_SLASSQ                                 815
#define FB_OP_DLASSQ                                 816
#define FB_OP_CLASSQ                                 817
#define FB_OP_ZLASSQ                                 818

#define FB_OP_SLASV2                                 819
#define FB_OP_DLASV2                                 820

#define FB_OP_SLASWP                                 821
#define FB_OP_DLASWP                                 822
#define FB_OP_CLASWP                                 823
#define FB_OP_ZLASWP                                 824

#define FB_OP_SLATBS                                 825
#define FB_OP_DLATBS                                 826
#define FB_OP_CLATBS                                 827
#define FB_OP_ZLATBS                                 828

#define FB_OP_SLATDF                                 829
#define FB_OP_DLATDF                                 830
#define FB_OP_CLATDF                                 831
#define FB_OP_ZLATDF                                 832

#define FB_OP_SLATPS                                 833
#define FB_OP_DLATPS                                 834
#define FB_OP_CLATPS                                 835
#define FB_OP_ZLATPS                                 836

#define FB_OP_SLATRD                                 837
#define FB_OP_DLATRD                                 838
#define FB_OP_CLATRD                                 839
#define FB_OP_ZLATRD                                 840

#define FB_OP_SLATRS                                 841
#define FB_OP_DLATRS                                 842
#define FB_OP_CLATRS                                 843
#define FB_OP_ZLATRS                                 844

#define FB_OP_SLATRZ                                 845
#define FB_OP_DLATRZ                                 846
#define FB_OP_CLATRZ                                 847
#define FB_OP_ZLATRZ                                 848

#define FB_OP_SLAUU2                                 849
#define FB_OP_DLAUU2                                 850
#define FB_OP_CLAUU2                                 851
#define FB_OP_ZLAUU2                                 852

#define FB_OP_SLAUUM                                 853
#define FB_OP_DLAUUM                                 854
#define FB_OP_CLAUUM                                 855
#define FB_OP_ZLAUUM                                 856

#define FB_OP_SLA_GBAMV                              857
#define FB_OP_DLA_GBAMV                              858
#define FB_OP_CLA_GBAMV                              859
#define FB_OP_ZLA_GBAMV                              860

#define FB_OP_CNRM2                                  861
#define FB_OP_ZNRM2                                  862

#define FB_OP_SOMATADD                               863
#define FB_OP_DOMATADD                               864
#define FB_OP_COMATADD                               865
#define FB_OP_ZOMATADD                               866

#define FB_OP_SOMATCOPY                              867
#define FB_OP_DOMATCOPY                              868
#define FB_OP_COMATCOPY                              869
#define FB_OP_ZOMATCOPY                              870

#define FB_OP_SOMATCOPY2                             871
#define FB_OP_DOMATCOPY2                             872
#define FB_OP_COMATCOPY2                             873
#define FB_OP_ZOMATCOPY2                             874

#define FB_OP_SORG2L                                 875
#define FB_OP_DORG2L                                 876

#define FB_OP_SORGBR                                 877
#define FB_OP_DORGBR                                 878

#define FB_OP_SORGHR                                 879
#define FB_OP_DORGHR                                 880

#define FB_OP_SORGL2                                 881
#define FB_OP_DORGL2                                 882

#define FB_OP_SORGLQ                                 883
#define FB_OP_DORGLQ                                 884

#define FB_OP_SORGQL                                 885
#define FB_OP_DORGQL                                 886

#define FB_OP_SORGQR                                 887
#define FB_OP_DORGQR                                 888

#define FB_OP_SORGR2                                 889
#define FB_OP_DORGR2                                 890

#define FB_OP_SORGRQ                                 891
#define FB_OP_DORGRQ                                 892

#define FB_OP_SORGTR                                 893
#define FB_OP_DORGTR                                 894

#define FB_OP_SORMBR                                 895
#define FB_OP_DORMBR                                 896

#define FB_OP_SORMHR                                 897
#define FB_OP_DORMHR                                 898

#define FB_OP_SORMLQ                                 899
#define FB_OP_DORMLQ                                 900

#define FB_OP_SORMQL                                 901
#define FB_OP_DORMQL                                 902

#define FB_OP_SORMQR                                 903
#define FB_OP_DORMQR                                 904

#define FB_OP_SORMR3                                 905
#define FB_OP_DORMR3                                 906

#define FB_OP_SORMRQ                                 907
#define FB_OP_DORMRQ                                 908

#define FB_OP_SORMRZ                                 909
#define FB_OP_DORMRZ                                 910

#define FB_OP_SPARSE                                 911

#define FB_OP_SPBCON                                 912
#define FB_OP_DPBCON                                 913
#define FB_OP_CPBCON                                 914
#define FB_OP_ZPBCON                                 915

#define FB_OP_SPBEQU                                 916
#define FB_OP_DPBEQU                                 917
#define FB_OP_CPBEQU                                 918
#define FB_OP_ZPBEQU                                 919

#define FB_OP_SPBRFS                                 920
#define FB_OP_DPBRFS                                 921
#define FB_OP_CPBRFS                                 922
#define FB_OP_ZPBRFS                                 923

#define FB_OP_SPBSV                                  924
#define FB_OP_DPBSV                                  925
#define FB_OP_CPBSV                                  926
#define FB_OP_ZPBSV                                  927

#define FB_OP_SPBSVX                                 928
#define FB_OP_DPBSVX                                 929
#define FB_OP_CPBSVX                                 930
#define FB_OP_ZPBSVX                                 931

#define FB_OP_SPBTRF                                 932
#define FB_OP_DPBTRF                                 933
#define FB_OP_CPBTRF                                 934
#define FB_OP_ZPBTRF                                 935

#define FB_OP_SPBTRS                                 936
#define FB_OP_DPBTRS                                 937
#define FB_OP_CPBTRS                                 938
#define FB_OP_ZPBTRS                                 939

#define FB_OP_SPOCON                                 940
#define FB_OP_DPOCON                                 941
#define FB_OP_CPOCON                                 942
#define FB_OP_ZPOCON                                 943

#define FB_OP_SPOEQU                                 944
#define FB_OP_DPOEQU                                 945
#define FB_OP_CPOEQU                                 946
#define FB_OP_ZPOEQU                                 947

#define FB_OP_SPORFS                                 948
#define FB_OP_DPORFS                                 949
#define FB_OP_CPORFS                                 950
#define FB_OP_ZPORFS                                 951

#define FB_OP_SPOSV                                  952
#define FB_OP_DPOSV                                  953
#define FB_OP_CPOSV                                  954
#define FB_OP_ZPOSV                                  955

#define FB_OP_SPOSVX                                 956
#define FB_OP_DPOSVX                                 957
#define FB_OP_CPOSVX                                 958
#define FB_OP_ZPOSVX                                 959

#define FB_OP_SPOSVXX                                960
#define FB_OP_DPOSVXX                                961
#define FB_OP_CPOSVXX                                962
#define FB_OP_ZPOSVXX                                963

#define FB_OP_SPOTRF                                 964
#define FB_OP_DPOTRF                                 965
#define FB_OP_CPOTRF                                 966
#define FB_OP_ZPOTRF                                 967

#define FB_OP_SPOTRI                                 968
#define FB_OP_DPOTRI                                 969
#define FB_OP_CPOTRI                                 970
#define FB_OP_ZPOTRI                                 971

#define FB_OP_SPOTRS                                 972
#define FB_OP_DPOTRS                                 973
#define FB_OP_CPOTRS                                 974
#define FB_OP_ZPOTRS                                 975

#define FB_OP_SPPCON                                 976
#define FB_OP_DPPCON                                 977
#define FB_OP_CPPCON                                 978
#define FB_OP_ZPPCON                                 979

#define FB_OP_SPPEQU                                 980
#define FB_OP_DPPEQU                                 981
#define FB_OP_CPPEQU                                 982
#define FB_OP_ZPPEQU                                 983

#define FB_OP_SPPRFS                                 984
#define FB_OP_DPPRFS                                 985
#define FB_OP_CPPRFS                                 986
#define FB_OP_ZPPRFS                                 987

#define FB_OP_SPPSV                                  988
#define FB_OP_DPPSV                                  989
#define FB_OP_CPPSV                                  990
#define FB_OP_ZPPSV                                  991

#define FB_OP_SPPSVX                                 992
#define FB_OP_DPPSVX                                 993
#define FB_OP_CPPSVX                                 994
#define FB_OP_ZPPSVX                                 995

#define FB_OP_SPPTRF                                 996
#define FB_OP_DPPTRF                                 997
#define FB_OP_CPPTRF                                 998
#define FB_OP_ZPPTRF                                 999

#define FB_OP_SPPTRI                                 1000
#define FB_OP_DPPTRI                                 1001
#define FB_OP_CPPTRI                                 1002
#define FB_OP_ZPPTRI                                 1003

#define FB_OP_SPPTRS                                 1004
#define FB_OP_DPPTRS                                 1005
#define FB_OP_CPPTRS                                 1006
#define FB_OP_ZPPTRS                                 1007

#define FB_OP_SPTCON                                 1008
#define FB_OP_DPTCON                                 1009
#define FB_OP_CPTCON                                 1010
#define FB_OP_ZPTCON                                 1011

#define FB_OP_SPTEQR                                 1012

#define FB_OP_SPTRFS                                 1013
#define FB_OP_DPTRFS                                 1014
#define FB_OP_CPTRFS                                 1015
#define FB_OP_ZPTRFS                                 1016

#define FB_OP_SPTSV                                  1017
#define FB_OP_DPTSV                                  1018
#define FB_OP_CPTSV                                  1019
#define FB_OP_ZPTSV                                  1020

#define FB_OP_SPTSVX                                 1021
#define FB_OP_DPTSVX                                 1022
#define FB_OP_CPTSVX                                 1023
#define FB_OP_ZPTSVX                                 1024

#define FB_OP_SPTTRF                                 1025
#define FB_OP_DPTTRF                                 1026
#define FB_OP_CPTTRF                                 1027
#define FB_OP_ZPTTRF                                 1028

#define FB_OP_SPTTRS                                 1029
#define FB_OP_DPTTRS                                 1030
#define FB_OP_CPTTRS                                 1031
#define FB_OP_ZPTTRS                                 1032

#define FB_OP_CROTG                                  1033
#define FB_OP_ZROTG                                  1034

#define FB_OP_SRSCL                                  1035
#define FB_OP_DRSCL                                  1036

#define FB_OP_SSBEV                                  1037
#define FB_OP_DSBEV                                  1038

#define FB_OP_SSBEVD                                 1039
#define FB_OP_DSBEVD                                 1040

#define FB_OP_SSBEVX                                 1041
#define FB_OP_DSBEVX                                 1042

#define FB_OP_SSBGV                                  1043
#define FB_OP_DSBGV                                  1044

#define FB_OP_SSBGVD                                 1045
#define FB_OP_DSBGVD                                 1046

#define FB_OP_SSBGVX                                 1047
#define FB_OP_DSBGVX                                 1048

#define FB_OP_CSBMV                                  1049
#define FB_OP_ZSBMV                                  1050

#define FB_OP_SSBTRF                                 1051
#define FB_OP_DSBTRF                                 1052

#define FB_OP_SSPCON                                 1053
#define FB_OP_DSPCON                                 1054
#define FB_OP_CSPCON                                 1055
#define FB_OP_ZSPCON                                 1056

#define FB_OP_SSPEV                                  1057
#define FB_OP_DSPEV                                  1058

#define FB_OP_SSPEVD                                 1059
#define FB_OP_DSPEVD                                 1060

#define FB_OP_SSPEVX                                 1061
#define FB_OP_DSPEVX                                 1062

#define FB_OP_SSPGV                                  1063
#define FB_OP_DSPGV                                  1064

#define FB_OP_SSPGVD                                 1065
#define FB_OP_DSPGVD                                 1066

#define FB_OP_SSPGVX                                 1067
#define FB_OP_DSPGVX                                 1068

#define FB_OP_CSPMV                                  1069
#define FB_OP_ZSPMV                                  1070

#define FB_OP_CSPR                                   1071
#define FB_OP_ZSPR                                   1072

#define FB_OP_CSPR2                                  1073
#define FB_OP_ZSPR2                                  1074

#define FB_OP_SSPRFS                                 1075
#define FB_OP_DSPRFS                                 1076
#define FB_OP_CSPRFS                                 1077
#define FB_OP_ZSPRFS                                 1078

#define FB_OP_SSPSV                                  1079
#define FB_OP_DSPSV                                  1080
#define FB_OP_CSPSV                                  1081
#define FB_OP_ZSPSV                                  1082

#define FB_OP_SSPSVX                                 1083
#define FB_OP_DSPSVX                                 1084

#define FB_OP_SSPTRF                                 1085
#define FB_OP_DSPTRF                                 1086
#define FB_OP_CSPTRF                                 1087
#define FB_OP_ZSPTRF                                 1088

#define FB_OP_SSPTRI                                 1089
#define FB_OP_DSPTRI                                 1090
#define FB_OP_CSPTRI                                 1091
#define FB_OP_ZSPTRI                                 1092

#define FB_OP_SSPTRS                                 1093
#define FB_OP_DSPTRS                                 1094
#define FB_OP_CSPTRS                                 1095
#define FB_OP_ZSPTRS                                 1096

#define FB_OP_CSROT                                  1097

#define FB_OP_CSRSCL                                 1098

#define FB_OP_SSTEBZ                                 1099
#define FB_OP_DSTEBZ                                 1100

#define FB_OP_SSTEDC                                 1101
#define FB_OP_DSTEDC                                 1102
#define FB_OP_CSTEDC                                 1103
#define FB_OP_ZSTEDC                                 1104

#define FB_OP_SSTEGR                                 1105
#define FB_OP_DSTEGR                                 1106
#define FB_OP_CSTEGR                                 1107
#define FB_OP_ZSTEGR                                 1108

#define FB_OP_SSTEIN                                 1109
#define FB_OP_DSTEIN                                 1110
#define FB_OP_CSTEIN                                 1111
#define FB_OP_ZSTEIN                                 1112

#define FB_OP_SSTEQR                                 1113
#define FB_OP_DSTEQR                                 1114
#define FB_OP_CSTEQR                                 1115
#define FB_OP_ZSTEQR                                 1116

#define FB_OP_SSTERF                                 1117
#define FB_OP_DSTERF                                 1118

#define FB_OP_SSTEV                                  1119
#define FB_OP_DSTEV                                  1120

#define FB_OP_SSTEVD                                 1121
#define FB_OP_DSTEVD                                 1122

#define FB_OP_SSTEVR                                 1123
#define FB_OP_DSTEVR                                 1124

#define FB_OP_SSTEVX                                 1125
#define FB_OP_DSTEVX                                 1126

#define FB_OP_SSYCON                                 1127
#define FB_OP_DSYCON                                 1128
#define FB_OP_CSYCON                                 1129
#define FB_OP_ZSYCON                                 1130

#define FB_OP_SSYEV                                  1131
#define FB_OP_DSYEV                                  1132

#define FB_OP_SSYEVD                                 1133
#define FB_OP_DSYEVD                                 1134

#define FB_OP_SSYEVR                                 1135
#define FB_OP_DSYEVR                                 1136

#define FB_OP_SSYEVX                                 1137
#define FB_OP_DSYEVX                                 1138

#define FB_OP_SSYGST                                 1139
#define FB_OP_DSYGST                                 1140
#define FB_OP_CSYGST                                 1141
#define FB_OP_ZSYGST                                 1142

#define FB_OP_SSYGV                                  1143
#define FB_OP_DSYGV                                  1144
#define FB_OP_CSYGV                                  1145
#define FB_OP_ZSYGV                                  1146

#define FB_OP_SSYGVD                                 1147
#define FB_OP_DSYGVD                                 1148
#define FB_OP_CSYGVD                                 1149
#define FB_OP_ZSYGVD                                 1150

#define FB_OP_SSYGVX                                 1151
#define FB_OP_DSYGVX                                 1152
#define FB_OP_CSYGVX                                 1153
#define FB_OP_ZSYGVX                                 1154

#define FB_OP_CSYMV                                  1155
#define FB_OP_ZSYMV                                  1156

#define FB_OP_CSYR                                   1157
#define FB_OP_ZSYR                                   1158

#define FB_OP_CSYR2                                  1159
#define FB_OP_ZSYR2                                  1160

#define FB_OP_SSYRFS                                 1161
#define FB_OP_DSYRFS                                 1162
#define FB_OP_CSYRFS                                 1163
#define FB_OP_ZSYRFS                                 1164

#define FB_OP_SSYSV                                  1165
#define FB_OP_DSYSV                                  1166
#define FB_OP_CSYSV                                  1167
#define FB_OP_ZSYSV                                  1168

#define FB_OP_SSYSVX                                 1169
#define FB_OP_DSYSVX                                 1170
#define FB_OP_CSYSVX                                 1171
#define FB_OP_ZSYSVX                                 1172

#define FB_OP_SSYSVXX                                1173
#define FB_OP_DSYSVXX                                1174
#define FB_OP_CSYSVXX                                1175
#define FB_OP_ZSYSVXX                                1176

#define FB_OP_SSYTRD                                 1177
#define FB_OP_DSYTRD                                 1178
#define FB_OP_CSYTRD                                 1179
#define FB_OP_ZSYTRD                                 1180

#define FB_OP_SSYTRF                                 1181
#define FB_OP_DSYTRF                                 1182
#define FB_OP_CSYTRF                                 1183
#define FB_OP_ZSYTRF                                 1184

#define FB_OP_SSYTRF_AA                              1185

#define FB_OP_SSYTRF_ROOK                            1186
#define FB_OP_DSYTRF_ROOK                            1187
#define FB_OP_CSYTRF_ROOK                            1188
#define FB_OP_ZSYTRF_ROOK                            1189

#define FB_OP_SSYTRI                                 1190
#define FB_OP_DSYTRI                                 1191
#define FB_OP_CSYTRI                                 1192
#define FB_OP_ZSYTRI                                 1193

#define FB_OP_SSYTRI2                                1194
#define FB_OP_DSYTRI2                                1195
#define FB_OP_CSYTRI2                                1196

#define FB_OP_SSYTRS                                 1197
#define FB_OP_DSYTRS                                 1198
#define FB_OP_CSYTRS                                 1199
#define FB_OP_ZSYTRS                                 1200

#define FB_OP_STBCON                                 1201
#define FB_OP_DTBCON                                 1202
#define FB_OP_CTBCON                                 1203
#define FB_OP_ZTBCON                                 1204

#define FB_OP_STBRFS                                 1205
#define FB_OP_DTBRFS                                 1206
#define FB_OP_CTBRFS                                 1207
#define FB_OP_ZTBRFS                                 1208

#define FB_OP_STBTRS                                 1209
#define FB_OP_DTBTRS                                 1210
#define FB_OP_CTBTRS                                 1211
#define FB_OP_ZTBTRS                                 1212

#define FB_OP_STGEVC                                 1213
#define FB_OP_DTGEVC                                 1214
#define FB_OP_CTGEVC                                 1215
#define FB_OP_ZTGEVC                                 1216

#define FB_OP_STGEX2                                 1217
#define FB_OP_DTGEX2                                 1218
#define FB_OP_CTGEX2                                 1219
#define FB_OP_ZTGEX2                                 1220

#define FB_OP_STGEXC                                 1221
#define FB_OP_DTGEXC                                 1222
#define FB_OP_CTGEXC                                 1223
#define FB_OP_ZTGEXC                                 1224

#define FB_OP_STGSEN                                 1225
#define FB_OP_DTGSEN                                 1226
#define FB_OP_CTGSEN                                 1227
#define FB_OP_ZTGSEN                                 1228

#define FB_OP_STGSJA                                 1229
#define FB_OP_DTGSJA                                 1230
#define FB_OP_CTGSJA                                 1231
#define FB_OP_ZTGSJA                                 1232

#define FB_OP_STGSNA                                 1233
#define FB_OP_DTGSNA                                 1234
#define FB_OP_CTGSNA                                 1235
#define FB_OP_ZTGSNA                                 1236

#define FB_OP_STGSY2                                 1237
#define FB_OP_DTGSY2                                 1238
#define FB_OP_CTGSY2                                 1239
#define FB_OP_ZTGSY2                                 1240

#define FB_OP_STGSYL                                 1241
#define FB_OP_DTGSYL                                 1242
#define FB_OP_CTGSYL                                 1243
#define FB_OP_ZTGSYL                                 1244

#define FB_OP_STPCON                                 1245
#define FB_OP_DTPCON                                 1246
#define FB_OP_CTPCON                                 1247
#define FB_OP_ZTPCON                                 1248

#define FB_OP_STPRFS                                 1249
#define FB_OP_DTPRFS                                 1250
#define FB_OP_CTPRFS                                 1251
#define FB_OP_ZTPRFS                                 1252

#define FB_OP_STPTRI                                 1253
#define FB_OP_DTPTRI                                 1254
#define FB_OP_CTPTRI                                 1255
#define FB_OP_ZTPTRI                                 1256

#define FB_OP_STPTRS                                 1257
#define FB_OP_DTPTRS                                 1258
#define FB_OP_CTPTRS                                 1259
#define FB_OP_ZTPTRS                                 1260

#define FB_OP_STRCON                                 1261
#define FB_OP_DTRCON                                 1262
#define FB_OP_CTRCON                                 1263
#define FB_OP_ZTRCON                                 1264

#define FB_OP_STREVC                                 1265
#define FB_OP_DTREVC                                 1266
#define FB_OP_CTREVC                                 1267
#define FB_OP_ZTREVC                                 1268

#define FB_OP_STRRFS                                 1269
#define FB_OP_DTRRFS                                 1270
#define FB_OP_CTRRFS                                 1271
#define FB_OP_ZTRRFS                                 1272

#define FB_OP_STRTRI                                 1273
#define FB_OP_DTRTRI                                 1274
#define FB_OP_CTRTRI                                 1275
#define FB_OP_ZTRTRI                                 1276

#define FB_OP_STRTRS                                 1277
#define FB_OP_DTRTRS                                 1278
#define FB_OP_CTRTRS                                 1279
#define FB_OP_ZTRTRS                                 1280

#define FB_OP_STZRZF                                 1281
#define FB_OP_DTZRZF                                 1282
#define FB_OP_CTZRZF                                 1283
#define FB_OP_ZTZRZF                                 1284

#define FB_OP_CUNG2L                                 1285
#define FB_OP_ZUNG2L                                 1286

#define FB_OP_CUNGBR                                 1287
#define FB_OP_ZUNGBR                                 1288

#define FB_OP_CUNGHR                                 1289
#define FB_OP_ZUNGHR                                 1290

#define FB_OP_CUNGL2                                 1291
#define FB_OP_ZUNGL2                                 1292

#define FB_OP_CUNGLQ                                 1293
#define FB_OP_ZUNGLQ                                 1294

#define FB_OP_CUNGQL                                 1295
#define FB_OP_ZUNGQL                                 1296

#define FB_OP_CUNGQR                                 1297
#define FB_OP_ZUNGQR                                 1298

#define FB_OP_CUNGR2                                 1299
#define FB_OP_ZUNGR2                                 1300

#define FB_OP_CUNGRQ                                 1301
#define FB_OP_ZUNGRQ                                 1302

#define FB_OP_CUNGTR                                 1303
#define FB_OP_ZUNGTR                                 1304

#define FB_OP_CUNMBR                                 1305
#define FB_OP_ZUNMBR                                 1306

#define FB_OP_CUNMHR                                 1307
#define FB_OP_ZUNMHR                                 1308

#define FB_OP_CUNMLQ                                 1309
#define FB_OP_ZUNMLQ                                 1310

#define FB_OP_CUNMQL                                 1311
#define FB_OP_ZUNMQL                                 1312

#define FB_OP_CUNMQR                                 1313
#define FB_OP_ZUNMQR                                 1314

#define FB_OP_CUNMR3                                 1315
#define FB_OP_ZUNMR3                                 1316

#define FB_OP_CUNMRQ                                 1317
#define FB_OP_ZUNMRQ                                 1318

#define FB_OP_CUNMRZ                                 1319
#define FB_OP_ZUNMRZ                                 1320

#define FB_OP_DZSUM1                                 1321

/* =========================================================================
 * ScaLAPACK — parallel p* distributed routines  (1322 – 1633)
 * ========================================================================= */

#define FB_OP_PCAMAX                                 1322

#define FB_OP_PCASUM                                 1323

#define FB_OP_PCAXPY                                 1324

#define FB_OP_PCCOPY                                 1325

#define FB_OP_PCDOT                                  1326

#define FB_OP_PCGBSV                                 1327

#define FB_OP_PCGBTRF                                1328

#define FB_OP_PCGBTRS                                1329

#define FB_OP_PCGEBRD                                1330

#define FB_OP_PCGECON                                1331

#define FB_OP_PCGEEQU                                1332

#define FB_OP_PCGEEVX                                1333

#define FB_OP_PCGEHRD                                1334

#define FB_OP_PCGELQF                                1335

#define FB_OP_PCGELS                                 1336

#define FB_OP_PCGELSS                                1337

#define FB_OP_PCGELSY                                1338

#define FB_OP_PCGEMM                                 1339

#define FB_OP_PCGEMV                                 1340

#define FB_OP_PCGEQLF                                1341

#define FB_OP_PCGER                                  1342

#define FB_OP_PCGERFS                                1343

#define FB_OP_PCGERQF                                1344

#define FB_OP_PCGESDD                                1345

#define FB_OP_PCGESV                                 1346

#define FB_OP_PCGESVD                                1347

#define FB_OP_PCGESVX                                1348

#define FB_OP_PCGETRF                                1349

#define FB_OP_PCGETRI                                1350

#define FB_OP_PCGETRS                                1351

#define FB_OP_PCGGHRD                                1352

#define FB_OP_PCHEEQUB                               1353

#define FB_OP_PCHEEV                                 1354

#define FB_OP_PCHEEVD                                1355

#define FB_OP_PCHEEVX                                1356

#define FB_OP_PCHEGST                                1357

#define FB_OP_PCHEGV                                 1358

#define FB_OP_PCHESV                                 1359

#define FB_OP_PCHETRD                                1360

#define FB_OP_PCHETRF                                1361

#define FB_OP_PCHETRS                                1362

#define FB_OP_PCNRM2                                 1363

#define FB_OP_PCPBSV                                 1364

#define FB_OP_PCPBTRF                                1365

#define FB_OP_PCPBTRS                                1366

#define FB_OP_PCPOCON                                1367

#define FB_OP_PCPOEQU                                1368

#define FB_OP_PCPORFS                                1369

#define FB_OP_PCPOSV                                 1370

#define FB_OP_PCPOSVX                                1371

#define FB_OP_PCPOTRF                                1372

#define FB_OP_PCPOTRI                                1373

#define FB_OP_PCPOTRS                                1374

#define FB_OP_PCPPSV                                 1375

#define FB_OP_PCPTSV                                 1376

#define FB_OP_PCPTTRF                                1377

#define FB_OP_PCPTTRS                                1378

#define FB_OP_PCSCAL                                 1379

#define FB_OP_PCSWAP                                 1380

#define FB_OP_PCSYGST                                1381

#define FB_OP_PCSYGV                                 1382

#define FB_OP_PCSYMM                                 1383

#define FB_OP_PCSYMV                                 1384

#define FB_OP_PCSYR                                  1385

#define FB_OP_PCSYR2                                 1386

#define FB_OP_PCSYR2K                                1387

#define FB_OP_PCSYRK                                 1388

#define FB_OP_PCSYSV                                 1389

#define FB_OP_PCSYTRD                                1390

#define FB_OP_PCSYTRF                                1391

#define FB_OP_PCSYTRS                                1392

#define FB_OP_PCTRCON                                1393

#define FB_OP_PCTRMM                                 1394

#define FB_OP_PCTRMV                                 1395

#define FB_OP_PCTRRFS                                1396

#define FB_OP_PCTRSM                                 1397

#define FB_OP_PCTRSV                                 1398

#define FB_OP_PCTRTRI                                1399

#define FB_OP_PCTZRZF                                1400

#define FB_OP_PCUNGQR                                1401

#define FB_OP_PCUNMQR                                1402

#define FB_OP_PDAMAX                                 1403

#define FB_OP_PDASUM                                 1404

#define FB_OP_PDAXPY                                 1405

#define FB_OP_PDCOPY                                 1406

#define FB_OP_PDDOT                                  1407

#define FB_OP_PDGBSV                                 1408

#define FB_OP_PDGBTRF                                1409

#define FB_OP_PDGBTRS                                1410

#define FB_OP_PDGEBRD                                1411

#define FB_OP_PDGECON                                1412

#define FB_OP_PDGEEQU                                1413

#define FB_OP_PDGEEVX                                1414

#define FB_OP_PDGEHRD                                1415

#define FB_OP_PDGELQF                                1416

#define FB_OP_PDGELS                                 1417

#define FB_OP_PDGELSS                                1418

#define FB_OP_PDGELSY                                1419

#define FB_OP_PDGEMM                                 1420

#define FB_OP_PDGEMV                                 1421

#define FB_OP_PDGEQLF                                1422

#define FB_OP_PDGER                                  1423

#define FB_OP_PDGERFS                                1424

#define FB_OP_PDGERQF                                1425

#define FB_OP_PDGESDD                                1426

#define FB_OP_PDGESV                                 1427

#define FB_OP_PDGESVD                                1428

#define FB_OP_PDGESVX                                1429

#define FB_OP_PDGETRF                                1430

#define FB_OP_PDGETRI                                1431

#define FB_OP_PDGETRS                                1432

#define FB_OP_PDGGHRD                                1433

#define FB_OP_PDNRM2                                 1434

#define FB_OP_PDORGQR                                1435

#define FB_OP_PDORMQR                                1436

#define FB_OP_PDPBSV                                 1437

#define FB_OP_PDPBTRF                                1438

#define FB_OP_PDPBTRS                                1439

#define FB_OP_PDPOCON                                1440

#define FB_OP_PDPOEQU                                1441

#define FB_OP_PDPORFS                                1442

#define FB_OP_PDPOSV                                 1443

#define FB_OP_PDPOSVX                                1444

#define FB_OP_PDPOTRF                                1445

#define FB_OP_PDPOTRI                                1446

#define FB_OP_PDPOTRS                                1447

#define FB_OP_PDPPSV                                 1448

#define FB_OP_PDPTSV                                 1449

#define FB_OP_PDPTTRF                                1450

#define FB_OP_PDPTTRS                                1451

#define FB_OP_PDSCAL                                 1452

#define FB_OP_PDSWAP                                 1453

#define FB_OP_PDSYEQUB                               1454

#define FB_OP_PDSYEV                                 1455

#define FB_OP_PDSYEVD                                1456

#define FB_OP_PDSYEVX                                1457

#define FB_OP_PDSYGST                                1458

#define FB_OP_PDSYGV                                 1459

#define FB_OP_PDSYMM                                 1460

#define FB_OP_PDSYMV                                 1461

#define FB_OP_PDSYR                                  1462

#define FB_OP_PDSYR2                                 1463

#define FB_OP_PDSYR2K                                1464

#define FB_OP_PDSYRK                                 1465

#define FB_OP_PDSYSV                                 1466

#define FB_OP_PDSYTRD                                1467

#define FB_OP_PDSYTRF                                1468

#define FB_OP_PDSYTRS                                1469

#define FB_OP_PDTRCON                                1470

#define FB_OP_PDTRMM                                 1471

#define FB_OP_PDTRMV                                 1472

#define FB_OP_PDTRRFS                                1473

#define FB_OP_PDTRSM                                 1474

#define FB_OP_PDTRSV                                 1475

#define FB_OP_PDTRTRI                                1476

#define FB_OP_PDTZRZF                                1477

#define FB_OP_PSAMAX                                 1478

#define FB_OP_PSASUM                                 1479

#define FB_OP_PSAXPY                                 1480

#define FB_OP_PSCOPY                                 1481

#define FB_OP_PSDOT                                  1482

#define FB_OP_PSGBSV                                 1483

#define FB_OP_PSGBTRF                                1484

#define FB_OP_PSGBTRS                                1485

#define FB_OP_PSGEBRD                                1486

#define FB_OP_PSGECON                                1487

#define FB_OP_PSGEEQU                                1488

#define FB_OP_PSGEEVX                                1489

#define FB_OP_PSGEHRD                                1490

#define FB_OP_PSGELQF                                1491

#define FB_OP_PSGELS                                 1492

#define FB_OP_PSGELSS                                1493

#define FB_OP_PSGELSY                                1494

#define FB_OP_PSGEMM                                 1495

#define FB_OP_PSGEMV                                 1496

#define FB_OP_PSGEQLF                                1497

#define FB_OP_PSGER                                  1498

#define FB_OP_PSGERFS                                1499

#define FB_OP_PSGERQF                                1500

#define FB_OP_PSGESDD                                1501

#define FB_OP_PSGESV                                 1502

#define FB_OP_PSGESVD                                1503

#define FB_OP_PSGESVX                                1504

#define FB_OP_PSGETRF                                1505

#define FB_OP_PSGETRI                                1506

#define FB_OP_PSGETRS                                1507

#define FB_OP_PSGGHRD                                1508

#define FB_OP_PSNRM2                                 1509

#define FB_OP_PSORGQR                                1510

#define FB_OP_PSORMQR                                1511

#define FB_OP_PSPBSV                                 1512

#define FB_OP_PSPBTRF                                1513

#define FB_OP_PSPBTRS                                1514

#define FB_OP_PSPOCON                                1515

#define FB_OP_PSPOEQU                                1516

#define FB_OP_PSPORFS                                1517

#define FB_OP_PSPOSV                                 1518

#define FB_OP_PSPOSVX                                1519

#define FB_OP_PSPOTRF                                1520

#define FB_OP_PSPOTRI                                1521

#define FB_OP_PSPOTRS                                1522

#define FB_OP_PSPPSV                                 1523

#define FB_OP_PSPTSV                                 1524

#define FB_OP_PSPTTRF                                1525

#define FB_OP_PSPTTRS                                1526

#define FB_OP_PSSCAL                                 1527

#define FB_OP_PSSWAP                                 1528

#define FB_OP_PSSYEQUB                               1529

#define FB_OP_PSSYEV                                 1530

#define FB_OP_PSSYEVD                                1531

#define FB_OP_PSSYEVX                                1532

#define FB_OP_PSSYGST                                1533

#define FB_OP_PSSYGV                                 1534

#define FB_OP_PSSYMM                                 1535

#define FB_OP_PSSYMV                                 1536

#define FB_OP_PSSYR                                  1537

#define FB_OP_PSSYR2                                 1538

#define FB_OP_PSSYR2K                                1539

#define FB_OP_PSSYRK                                 1540

#define FB_OP_PSSYSV                                 1541

#define FB_OP_PSSYTRD                                1542

#define FB_OP_PSSYTRF                                1543

#define FB_OP_PSSYTRS                                1544

#define FB_OP_PSTRCON                                1545

#define FB_OP_PSTRMM                                 1546

#define FB_OP_PSTRMV                                 1547

#define FB_OP_PSTRRFS                                1548

#define FB_OP_PSTRSM                                 1549

#define FB_OP_PSTRSV                                 1550

#define FB_OP_PSTRTRI                                1551

#define FB_OP_PSTZRZF                                1552

#define FB_OP_PZAMAX                                 1553

#define FB_OP_PZASUM                                 1554

#define FB_OP_PZAXPY                                 1555

#define FB_OP_PZCOPY                                 1556

#define FB_OP_PZDOT                                  1557

#define FB_OP_PZGBSV                                 1558

#define FB_OP_PZGBTRF                                1559

#define FB_OP_PZGBTRS                                1560

#define FB_OP_PZGEBRD                                1561

#define FB_OP_PZGECON                                1562

#define FB_OP_PZGEEQU                                1563

#define FB_OP_PZGEEVX                                1564

#define FB_OP_PZGEHRD                                1565

#define FB_OP_PZGELQF                                1566

#define FB_OP_PZGELS                                 1567

#define FB_OP_PZGELSS                                1568

#define FB_OP_PZGELSY                                1569

#define FB_OP_PZGEMM                                 1570

#define FB_OP_PZGEMV                                 1571

#define FB_OP_PZGEQLF                                1572

#define FB_OP_PZGER                                  1573

#define FB_OP_PZGERFS                                1574

#define FB_OP_PZGERQF                                1575

#define FB_OP_PZGESDD                                1576

#define FB_OP_PZGESV                                 1577

#define FB_OP_PZGESVD                                1578

#define FB_OP_PZGESVX                                1579

#define FB_OP_PZGETRF                                1580

#define FB_OP_PZGETRI                                1581

#define FB_OP_PZGETRS                                1582

#define FB_OP_PZGGHRD                                1583

#define FB_OP_PZHEEQUB                               1584

#define FB_OP_PZHEEV                                 1585

#define FB_OP_PZHEEVD                                1586

#define FB_OP_PZHEEVX                                1587

#define FB_OP_PZHEGST                                1588

#define FB_OP_PZHEGV                                 1589

#define FB_OP_PZHESV                                 1590

#define FB_OP_PZHETRD                                1591

#define FB_OP_PZHETRF                                1592

#define FB_OP_PZHETRS                                1593

#define FB_OP_PZNRM2                                 1594

#define FB_OP_PZPBSV                                 1595

#define FB_OP_PZPBTRF                                1596

#define FB_OP_PZPBTRS                                1597

#define FB_OP_PZPOCON                                1598

#define FB_OP_PZPOEQU                                1599

#define FB_OP_PZPORFS                                1600

#define FB_OP_PZPOSV                                 1601

#define FB_OP_PZPOSVX                                1602

#define FB_OP_PZPOTRF                                1603

#define FB_OP_PZPOTRI                                1604

#define FB_OP_PZPOTRS                                1605

#define FB_OP_PZPPSV                                 1606

#define FB_OP_PZPTSV                                 1607

#define FB_OP_PZPTTRF                                1608

#define FB_OP_PZPTTRS                                1609

#define FB_OP_PZSCAL                                 1610

#define FB_OP_PZSWAP                                 1611

#define FB_OP_PZSYGST                                1612

#define FB_OP_PZSYGV                                 1613

#define FB_OP_PZSYMM                                 1614

#define FB_OP_PZSYMV                                 1615

#define FB_OP_PZSYR                                  1616

#define FB_OP_PZSYR2                                 1617

#define FB_OP_PZSYR2K                                1618

#define FB_OP_PZSYRK                                 1619

#define FB_OP_PZSYSV                                 1620

#define FB_OP_PZSYTRD                                1621

#define FB_OP_PZSYTRF                                1622

#define FB_OP_PZSYTRS                                1623

#define FB_OP_PZTRCON                                1624

#define FB_OP_PZTRMM                                 1625

#define FB_OP_PZTRMV                                 1626

#define FB_OP_PZTRRFS                                1627

#define FB_OP_PZTRSM                                 1628

#define FB_OP_PZTRSV                                 1629

#define FB_OP_PZTRTRI                                1630

#define FB_OP_PZTZRZF                                1631

#define FB_OP_PZUNGQR                                1632

#define FB_OP_PZUNMQR                                1633

/* =========================================================================
 * Extended BLAS (cblas_batch / cblas_strided / axpby variants)  (1634 – 1807)
 * ========================================================================= */

#define FB_OP_SAXPBY                                 1634
#define FB_OP_DAXPBY                                 1635
#define FB_OP_CAXPBY                                 1636
#define FB_OP_ZAXPBY                                 1637

#define FB_OP_SAXPY_BATCH                            1638
#define FB_OP_DAXPY_BATCH                            1639
#define FB_OP_CAXPY_BATCH                            1640
#define FB_OP_ZAXPY_BATCH                            1641

#define FB_OP_SAXPY_BATCH_STRIDED                    1642
#define FB_OP_DAXPY_BATCH_STRIDED                    1643
#define FB_OP_CAXPY_BATCH_STRIDED                    1644
#define FB_OP_ZAXPY_BATCH_STRIDED                    1645

#define FB_OP_CBLAS_CAXPBY                           1646

#define FB_OP_CBLAS_CAXPY_BATCH                      1647

#define FB_OP_CBLAS_CAXPY_BATCH_STRIDED              1648

#define FB_OP_CBLAS_CCOPY_BATCH                      1649

#define FB_OP_CBLAS_CCOPY_BATCH_STRIDED              1650

#define FB_OP_CBLAS_CDGMM_BATCH                      1651

#define FB_OP_CBLAS_CDGMM_BATCH_STRIDED              1652

#define FB_OP_CBLAS_CGEMM3M_BATCH                    1653

#define FB_OP_CBLAS_CGEMM3M_BATCH_STRIDED            1654

#define FB_OP_CBLAS_CGEMMT                           1655

#define FB_OP_CBLAS_CGEMM_BATCH                      1656

#define FB_OP_CBLAS_CGEMM_BATCH_STRIDED              1657

#define FB_OP_CBLAS_CGEMM_COMPUTE                    1658

#define FB_OP_CBLAS_CGEMM_PACK                       1659

#define FB_OP_CBLAS_CGEMM_PACK_GET_SIZE              1660

#define FB_OP_CBLAS_CGEMV_BATCH                      1661

#define FB_OP_CBLAS_CGEMV_BATCH_STRIDED              1662

#define FB_OP_CBLAS_CSYMM_BATCH                      1663

#define FB_OP_CBLAS_CSYR2K_BATCH                     1664

#define FB_OP_CBLAS_CSYRK_BATCH                      1665

#define FB_OP_CBLAS_CTRSM_BATCH                      1666

#define FB_OP_CBLAS_CTRSM_BATCH_STRIDED              1667

#define FB_OP_CBLAS_DAXPBY                           1668

#define FB_OP_CBLAS_DAXPY_BATCH                      1669

#define FB_OP_CBLAS_DAXPY_BATCH_STRIDED              1670

#define FB_OP_CBLAS_DCOPY_BATCH                      1671

#define FB_OP_CBLAS_DCOPY_BATCH_STRIDED              1672

#define FB_OP_CBLAS_DDGMM_BATCH                      1673

#define FB_OP_CBLAS_DDGMM_BATCH_STRIDED              1674

#define FB_OP_CBLAS_DGEMM3M_BATCH                    1675

#define FB_OP_CBLAS_DGEMM3M_BATCH_STRIDED            1676

#define FB_OP_CBLAS_DGEMMT                           1677

#define FB_OP_CBLAS_DGEMM_BATCH                      1678

#define FB_OP_CBLAS_DGEMM_BATCH_STRIDED              1679

#define FB_OP_CBLAS_DGEMM_COMPUTE                    1680

#define FB_OP_CBLAS_DGEMM_PACK                       1681

#define FB_OP_CBLAS_DGEMM_PACK_GET_SIZE              1682

#define FB_OP_CBLAS_DGEMV_BATCH                      1683

#define FB_OP_CBLAS_DGEMV_BATCH_STRIDED              1684

#define FB_OP_CBLAS_DSYMM_BATCH                      1685

#define FB_OP_CBLAS_DSYR2K_BATCH                     1686

#define FB_OP_CBLAS_DSYRK_BATCH                      1687

#define FB_OP_CBLAS_DTRSM_BATCH                      1688

#define FB_OP_CBLAS_DTRSM_BATCH_STRIDED              1689

#define FB_OP_CBLAS_GEMM_BF16BF16F32                 1690

#define FB_OP_CBLAS_GEMM_E4M3E4M3F32                 1691

#define FB_OP_CBLAS_GEMM_E5M2E5M2F32                 1692

#define FB_OP_CBLAS_GEMM_F16F16F32                   1693

#define FB_OP_CBLAS_GEMM_S8S8S32                     1694

#define FB_OP_CBLAS_GEMM_S8U8S32                     1695

#define FB_OP_CBLAS_SAXPBY                           1696

#define FB_OP_CBLAS_SAXPY_BATCH                      1697

#define FB_OP_CBLAS_SAXPY_BATCH_STRIDED              1698

#define FB_OP_CBLAS_SCOPY_BATCH                      1699

#define FB_OP_CBLAS_SCOPY_BATCH_STRIDED              1700

#define FB_OP_CBLAS_SDGMM_BATCH                      1701

#define FB_OP_CBLAS_SDGMM_BATCH_STRIDED              1702

#define FB_OP_CBLAS_SGEMM3M_BATCH                    1703

#define FB_OP_CBLAS_SGEMM3M_BATCH_STRIDED            1704

#define FB_OP_CBLAS_SGEMMT                           1705

#define FB_OP_CBLAS_SGEMM_BATCH                      1706

#define FB_OP_CBLAS_SGEMM_BATCH_STRIDED              1707

#define FB_OP_CBLAS_SGEMM_COMPUTE                    1708

#define FB_OP_CBLAS_SGEMM_PACK                       1709

#define FB_OP_CBLAS_SGEMM_PACK_GET_SIZE              1710

#define FB_OP_CBLAS_SGEMV_BATCH                      1711

#define FB_OP_CBLAS_SGEMV_BATCH_STRIDED              1712

#define FB_OP_CBLAS_SSYMM_BATCH                      1713

#define FB_OP_CBLAS_SSYR2K_BATCH                     1714

#define FB_OP_CBLAS_SSYRK_BATCH                      1715

#define FB_OP_CBLAS_STRSM_BATCH                      1716

#define FB_OP_CBLAS_STRSM_BATCH_STRIDED              1717

#define FB_OP_CBLAS_ZAXPBY                           1718

#define FB_OP_CBLAS_ZAXPY_BATCH                      1719

#define FB_OP_CBLAS_ZAXPY_BATCH_STRIDED              1720

#define FB_OP_CBLAS_ZCOPY_BATCH                      1721

#define FB_OP_CBLAS_ZCOPY_BATCH_STRIDED              1722

#define FB_OP_CBLAS_ZDGMM_BATCH                      1723

#define FB_OP_CBLAS_ZDGMM_BATCH_STRIDED              1724

#define FB_OP_CBLAS_ZGEMM3M_BATCH                    1725

#define FB_OP_CBLAS_ZGEMM3M_BATCH_STRIDED            1726

#define FB_OP_CBLAS_ZGEMMT                           1727

#define FB_OP_CBLAS_ZGEMM_BATCH                      1728

#define FB_OP_CBLAS_ZGEMM_BATCH_STRIDED              1729

#define FB_OP_CBLAS_ZGEMM_COMPUTE                    1730

#define FB_OP_CBLAS_ZGEMM_PACK                       1731

#define FB_OP_CBLAS_ZGEMM_PACK_GET_SIZE              1732

#define FB_OP_CBLAS_ZGEMV_BATCH                      1733

#define FB_OP_CBLAS_ZGEMV_BATCH_STRIDED              1734

#define FB_OP_CBLAS_ZSYMM_BATCH                      1735

#define FB_OP_CBLAS_ZSYR2K_BATCH                     1736

#define FB_OP_CBLAS_ZSYRK_BATCH                      1737

#define FB_OP_CBLAS_ZTRSM_BATCH                      1738

#define FB_OP_CBLAS_ZTRSM_BATCH_STRIDED              1739

#define FB_OP_SCOPY_BATCH                            1740
#define FB_OP_DCOPY_BATCH                            1741
#define FB_OP_CCOPY_BATCH                            1742
#define FB_OP_ZCOPY_BATCH                            1743

#define FB_OP_SCOPY_BATCH_STRIDED                    1744
#define FB_OP_DCOPY_BATCH_STRIDED                    1745
#define FB_OP_CCOPY_BATCH_STRIDED                    1746
#define FB_OP_ZCOPY_BATCH_STRIDED                    1747

#define FB_OP_SDGMM_BATCH                            1748
#define FB_OP_DDGMM_BATCH                            1749
#define FB_OP_CDGMM_BATCH                            1750
#define FB_OP_ZDGMM_BATCH                            1751

#define FB_OP_SDGMM_BATCH_STRIDED                    1752
#define FB_OP_DDGMM_BATCH_STRIDED                    1753
#define FB_OP_CDGMM_BATCH_STRIDED                    1754
#define FB_OP_ZDGMM_BATCH_STRIDED                    1755

#define FB_OP_SGEMM3M_BATCH                          1756
#define FB_OP_DGEMM3M_BATCH                          1757
#define FB_OP_CGEMM3M_BATCH                          1758
#define FB_OP_ZGEMM3M_BATCH                          1759

#define FB_OP_SGEMM3M_BATCH_STRIDED                  1760
#define FB_OP_DGEMM3M_BATCH_STRIDED                  1761
#define FB_OP_CGEMM3M_BATCH_STRIDED                  1762
#define FB_OP_ZGEMM3M_BATCH_STRIDED                  1763

#define FB_OP_SGEMV_BATCH                            1764
#define FB_OP_DGEMV_BATCH                            1765
#define FB_OP_CGEMV_BATCH                            1766
#define FB_OP_ZGEMV_BATCH                            1767

#define FB_OP_SGEMV_BATCH_STRIDED                    1768
#define FB_OP_DGEMV_BATCH_STRIDED                    1769
#define FB_OP_CGEMV_BATCH_STRIDED                    1770
#define FB_OP_ZGEMV_BATCH_STRIDED                    1771

#define FB_OP_SIMATCOPY_BATCH                        1772
#define FB_OP_DIMATCOPY_BATCH                        1773
#define FB_OP_CIMATCOPY_BATCH                        1774
#define FB_OP_ZIMATCOPY_BATCH                        1775

#define FB_OP_SIMATCOPY_BATCH_STRIDED                1776
#define FB_OP_DIMATCOPY_BATCH_STRIDED                1777
#define FB_OP_CIMATCOPY_BATCH_STRIDED                1778
#define FB_OP_ZIMATCOPY_BATCH_STRIDED                1779

#define FB_OP_SOMATCOPY_BATCH                        1780
#define FB_OP_DOMATCOPY_BATCH                        1781
#define FB_OP_COMATCOPY_BATCH                        1782
#define FB_OP_ZOMATCOPY_BATCH                        1783

#define FB_OP_SOMATCOPY_BATCH_STRIDED                1784
#define FB_OP_DOMATCOPY_BATCH_STRIDED                1785
#define FB_OP_COMATCOPY_BATCH_STRIDED                1786
#define FB_OP_ZOMATCOPY_BATCH_STRIDED                1787

#define FB_OP_SSYMM_BATCH                            1788
#define FB_OP_DSYMM_BATCH                            1789
#define FB_OP_CSYMM_BATCH                            1790
#define FB_OP_ZSYMM_BATCH                            1791

#define FB_OP_SSYR2K_BATCH                           1792
#define FB_OP_DSYR2K_BATCH                           1793
#define FB_OP_CSYR2K_BATCH                           1794
#define FB_OP_ZSYR2K_BATCH                           1795

#define FB_OP_SSYRK_BATCH                            1796
#define FB_OP_DSYRK_BATCH                            1797
#define FB_OP_CSYRK_BATCH                            1798
#define FB_OP_ZSYRK_BATCH                            1799

#define FB_OP_STRSM_BATCH                            1800
#define FB_OP_DTRSM_BATCH                            1801
#define FB_OP_CTRSM_BATCH                            1802
#define FB_OP_ZTRSM_BATCH                            1803

#define FB_OP_STRSM_BATCH_STRIDED                    1804
#define FB_OP_DTRSM_BATCH_STRIDED                    1805
#define FB_OP_CTRSM_BATCH_STRIDED                    1806
#define FB_OP_ZTRSM_BATCH_STRIDED                    1807

/* =========================================================================
 * MKL extensions (mkl_jit, mkl_*omatcopy, mkl_sparse_*)  (1808 – 1850)
 * ========================================================================= */

#define FB_OP_MKL_CIMATCOPY                          1808

#define FB_OP_MKL_CIMATCOPY_BATCH                    1809

#define FB_OP_MKL_CIMATCOPY_BATCH_STRIDED            1810

#define FB_OP_MKL_COMATADD                           1811

#define FB_OP_MKL_COMATCOPY                          1812

#define FB_OP_MKL_COMATCOPY2                         1813

#define FB_OP_MKL_COMATCOPY_BATCH                    1814

#define FB_OP_MKL_COMATCOPY_BATCH_STRIDED            1815

#define FB_OP_MKL_DIMATCOPY                          1816

#define FB_OP_MKL_DIMATCOPY_BATCH                    1817

#define FB_OP_MKL_DIMATCOPY_BATCH_STRIDED            1818

#define FB_OP_MKL_DOMATADD                           1819

#define FB_OP_MKL_DOMATCOPY                          1820

#define FB_OP_MKL_DOMATCOPY2                         1821

#define FB_OP_MKL_DOMATCOPY_BATCH                    1822

#define FB_OP_MKL_DOMATCOPY_BATCH_STRIDED            1823

#define FB_OP_MKL_JIT_CREATE_CGEMM                   1824

#define FB_OP_MKL_JIT_CREATE_DGEMM                   1825

#define FB_OP_MKL_JIT_CREATE_SGEMM                   1826

#define FB_OP_MKL_JIT_CREATE_ZGEMM                   1827

#define FB_OP_MKL_JIT_DESTROY                        1828

#define FB_OP_MKL_JIT_GET_CGEMM_PTR                  1829

#define FB_OP_MKL_JIT_GET_DGEMM_PTR                  1830

#define FB_OP_MKL_JIT_GET_SGEMM_PTR                  1831

#define FB_OP_MKL_JIT_GET_ZGEMM_PTR                  1832

#define FB_OP_MKL_SIMATCOPY                          1833

#define FB_OP_MKL_SIMATCOPY_BATCH                    1834

#define FB_OP_MKL_SIMATCOPY_BATCH_STRIDED            1835

#define FB_OP_MKL_SOMATADD                           1836

#define FB_OP_MKL_SOMATCOPY                          1837

#define FB_OP_MKL_SOMATCOPY2                         1838

#define FB_OP_MKL_SOMATCOPY_BATCH                    1839

#define FB_OP_MKL_SOMATCOPY_BATCH_STRIDED            1840

#define FB_OP_MKL_SPARSE_S_CREATE_CSR                1841

#define FB_OP_MKL_SPARSE_S_MV                        1842

#define FB_OP_MKL_ZIMATCOPY                          1843

#define FB_OP_MKL_ZIMATCOPY_BATCH                    1844

#define FB_OP_MKL_ZIMATCOPY_BATCH_STRIDED            1845

#define FB_OP_MKL_ZOMATADD                           1846

#define FB_OP_MKL_ZOMATCOPY                          1847

#define FB_OP_MKL_ZOMATCOPY2                         1848

#define FB_OP_MKL_ZOMATCOPY_BATCH                    1849

#define FB_OP_MKL_ZOMATCOPY_BATCH_STRIDED            1850

/* =========================================================================
 * Deep Neural Network primitives (fb_dnn_*)  (1851 – 1903)
 * ========================================================================= */

#define FB_OP_FB_DNN_ADAPTIVEPOOL                    1851

#define FB_OP_FB_DNN_AVGPOOL_BACKWARD                1852

#define FB_OP_FB_DNN_AVGPOOL_FORWARD                 1853

#define FB_OP_FB_DNN_BATCHNORM_BACKWARD              1854

#define FB_OP_FB_DNN_BATCHNORM_FORWARD_INFERENCE     1855

#define FB_OP_FB_DNN_BATCHNORM_FORWARD_TRAINING      1856

#define FB_OP_FB_DNN_CONV2D_FORWARD                  1857

#define FB_OP_FB_DNN_CONV3D_FORWARD                  1858

#define FB_OP_FB_DNN_CONV_BACKWARD_BIAS              1859

#define FB_OP_FB_DNN_CONV_BACKWARD_DATA              1860

#define FB_OP_FB_DNN_CONV_BACKWARD_FILTER            1861

#define FB_OP_FB_DNN_CONV_BATCHNORM_RELU             1862

#define FB_OP_FB_DNN_CONV_BIAS_RELU                  1863

#define FB_OP_FB_DNN_CONV_DEPTHWISE                  1864

#define FB_OP_FB_DNN_CONV_DILATED                    1865

#define FB_OP_FB_DNN_CONV_FIND_ALGORITHM             1866

#define FB_OP_FB_DNN_CONV_GROUPED                    1867

#define FB_OP_FB_DNN_CONV_TRANSPOSED                 1868

#define FB_OP_FB_DNN_DROPOUT_BACKWARD                1869

#define FB_OP_FB_DNN_DROPOUT_FORWARD                 1870

#define FB_OP_FB_DNN_ELU                             1871

#define FB_OP_FB_DNN_FLASH_ATTENTION_BACKWARD        1872

#define FB_OP_FB_DNN_FLASH_ATTENTION_FORWARD         1873

#define FB_OP_FB_DNN_GELU                            1874

#define FB_OP_FB_DNN_GLOBALPOOL                      1875

#define FB_OP_FB_DNN_GROUPNORM_FORWARD               1876

#define FB_OP_FB_DNN_GRU_FORWARD                     1877

#define FB_OP_FB_DNN_INSTANCENORM_FORWARD            1878

#define FB_OP_FB_DNN_LAYERNORM_BACKWARD              1879

#define FB_OP_FB_DNN_LAYERNORM_FORWARD               1880

#define FB_OP_FB_DNN_LEAKY_RELU                      1881

#define FB_OP_FB_DNN_LINEAR_GELU                     1882

#define FB_OP_FB_DNN_LINEAR_RELU                     1883

#define FB_OP_FB_DNN_LOGSOFTMAX_BACKWARD             1884

#define FB_OP_FB_DNN_LOGSOFTMAX_FORWARD              1885

#define FB_OP_FB_DNN_LSTM_FORWARD                    1886

#define FB_OP_FB_DNN_MAXPOOL_BACKWARD                1887

#define FB_OP_FB_DNN_MAXPOOL_FORWARD                 1888

#define FB_OP_FB_DNN_MISH                            1889

#define FB_OP_FB_DNN_MULTI_HEAD_ATTENTION            1890

#define FB_OP_FB_DNN_RELU                            1891

#define FB_OP_FB_DNN_RELU_BACKWARD                   1892

#define FB_OP_FB_DNN_RESIDUAL_BLOCK                  1893

#define FB_OP_FB_DNN_RNN_BACKWARD_DATA               1894

#define FB_OP_FB_DNN_RNN_BACKWARD_WEIGHTS            1895

#define FB_OP_FB_DNN_RNN_FORWARD                     1896

#define FB_OP_FB_DNN_SCALED_DOT_PRODUCT_ATTENTION    1897

#define FB_OP_FB_DNN_SIGMOID                         1898

#define FB_OP_FB_DNN_SOFTMAX_BACKWARD                1899

#define FB_OP_FB_DNN_SOFTMAX_FORWARD                 1900

#define FB_OP_FB_DNN_SOFTPLUS                        1901

#define FB_OP_FB_DNN_SWISH                           1902

#define FB_OP_FB_DNN_TANH                            1903

/* =========================================================================
 * FFT (fb_fft_*)  (1904 – 1942)
 * ========================================================================= */

#define FB_OP_FB_FFT_CLEAR_CALLBACKS                 1904

#define FB_OP_FB_FFT_COMMIT                          1905

#define FB_OP_FB_FFT_CREATE_PLAN_1D                  1906

#define FB_OP_FB_FFT_CREATE_PLAN_2D                  1907

#define FB_OP_FB_FFT_CREATE_PLAN_3D                  1908

#define FB_OP_FB_FFT_CREATE_PLAN_MANY                1909

#define FB_OP_FB_FFT_DESTROY                         1910

#define FB_OP_FB_FFT_ESTIMATE_WORKSPACE_1D           1911

#define FB_OP_FB_FFT_EXECUTE_BACKWARD_C2C            1912

#define FB_OP_FB_FFT_EXECUTE_BACKWARD_C2R            1913

#define FB_OP_FB_FFT_EXECUTE_BACKWARD_Z2D            1914

#define FB_OP_FB_FFT_EXECUTE_BACKWARD_Z2Z            1915

#define FB_OP_FB_FFT_EXECUTE_FORWARD_C2C             1916

#define FB_OP_FB_FFT_EXECUTE_FORWARD_D2Z             1917

#define FB_OP_FB_FFT_EXECUTE_FORWARD_R2C             1918

#define FB_OP_FB_FFT_EXECUTE_FORWARD_Z2Z             1919

#define FB_OP_FB_FFT_EXECUTE_MULTI_GPU               1920

#define FB_OP_FB_FFT_EXECUTE_ROUND_TRIP_D2Z2D        1921

#define FB_OP_FB_FFT_EXECUTE_ROUND_TRIP_R2C2R        1922

#define FB_OP_FB_FFT_FREE_DISTRIBUTED                1923

#define FB_OP_FB_FFT_GET_BACKEND_NAME                1924

#define FB_OP_FB_FFT_GET_MAX_DIMENSIONS              1925

#define FB_OP_FB_FFT_GET_OPTIMAL_SIZE                1926

#define FB_OP_FB_FFT_GET_SUPPORTED_TRANSFORMS        1927

#define FB_OP_FB_FFT_GET_VERSION                     1928

#define FB_OP_FB_FFT_GET_WORKSPACE_SIZE              1929

#define FB_OP_FB_FFT_IS_SIZE_OPTIMAL                 1930

#define FB_OP_FB_FFT_MALLOC_DISTRIBUTED              1931

#define FB_OP_FB_FFT_PLAN                            1932

#define FB_OP_FB_FFT_SET_GPUS                        1933

#define FB_OP_FB_FFT_SET_LOAD_CALLBACK               1934

#define FB_OP_FB_FFT_SET_NORMALIZATION               1935

#define FB_OP_FB_FFT_SET_PARAMETER                   1936

#define FB_OP_FB_FFT_SET_PLACEMENT                   1937

#define FB_OP_FB_FFT_SET_SCALE                       1938

#define FB_OP_FB_FFT_SET_STORE_CALLBACK              1939

#define FB_OP_FB_FFT_SET_STREAM                      1940

#define FB_OP_FB_FFT_SET_STRIDE                      1941

#define FB_OP_FB_FFT_WAIT_STREAM                     1942

/* =========================================================================
 * Sparse linear algebra (fb_sparse_*)  (1943 – 2000)
 * ========================================================================= */

#define FB_OP_FB_SPARSE_ADD                          1943

#define FB_OP_FB_SPARSE_BICG                         1944

#define FB_OP_FB_SPARSE_BICGSTAB                     1945

#define FB_OP_FB_SPARSE_CG                           1946

#define FB_OP_FB_SPARSE_CONVERT_BSR                  1947

#define FB_OP_FB_SPARSE_CONVERT_COO                  1948

#define FB_OP_FB_SPARSE_CONVERT_CSR                  1949

#define FB_OP_FB_SPARSE_COPY                         1950

#define FB_OP_FB_SPARSE_CREATE_BSR                   1951

#define FB_OP_FB_SPARSE_CREATE_COO                   1952

#define FB_OP_FB_SPARSE_CREATE_CSC                   1953

#define FB_OP_FB_SPARSE_CREATE_CSR                   1954

#define FB_OP_FB_SPARSE_DESTROY                      1955

#define FB_OP_FB_SPARSE_EXPORT                       1956

#define FB_OP_FB_SPARSE_FGMRES                       1957

#define FB_OP_FB_SPARSE_GET_FORMAT                   1958

#define FB_OP_FB_SPARSE_GET_SIZE                     1959

#define FB_OP_FB_SPARSE_GMRES                        1960

#define FB_OP_FB_SPARSE_HERMM                        1961

#define FB_OP_FB_SPARSE_HERMV                        1962

#define FB_OP_FB_SPARSE_MINRES                       1963

#define FB_OP_FB_SPARSE_MM                           1964

#define FB_OP_FB_SPARSE_MV                           1965

#define FB_OP_FB_SPARSE_OPTIMIZE                     1966

#define FB_OP_FB_SPARSE_PRECOND_APPLY                1967

#define FB_OP_FB_SPARSE_PRECOND_GS                   1968

#define FB_OP_FB_SPARSE_PRECOND_IC0_CREATE           1969

#define FB_OP_FB_SPARSE_PRECOND_ILU0_CREATE          1970

#define FB_OP_FB_SPARSE_PRECOND_ILUP_CREATE          1971

#define FB_OP_FB_SPARSE_PRECOND_JACOBI_CREATE        1972

#define FB_OP_FB_SPARSE_PRECOND_TRSV_LOWER           1973

#define FB_OP_FB_SPARSE_PRUNE_TO_2_4                 1974

#define FB_OP_FB_SPARSE_QMR                          1975

#define FB_OP_FB_SPARSE_SET_DIAG_TYPE                1976

#define FB_OP_FB_SPARSE_SET_FILL_MODE                1977

#define FB_OP_FB_SPARSE_SET_MATRIX_TYPE              1978

#define FB_OP_FB_SPARSE_SET_MM_HINT                  1979

#define FB_OP_FB_SPARSE_SET_MV_HINT                  1980

#define FB_OP_FB_SPARSE_SET_SV_HINT                  1981

#define FB_OP_FB_SPARSE_SOLVER_DESTROY               1982

#define FB_OP_FB_SPARSE_SOLVER_GET_ITERS             1983

#define FB_OP_FB_SPARSE_SOLVER_GET_RESIDUAL          1984

#define FB_OP_FB_SPARSE_SOLVER_INIT                  1985

#define FB_OP_FB_SPARSE_SOLVER_SET_MAXITER           1986

#define FB_OP_FB_SPARSE_SOLVER_SET_PRECOND           1987

#define FB_OP_FB_SPARSE_SOLVER_SET_TOL               1988

#define FB_OP_FB_SPARSE_SOLVER_SOLVE                 1989

#define FB_OP_FB_SPARSE_SPMM                         1990

#define FB_OP_FB_SPARSE_SPMV                         1991

#define FB_OP_FB_SPARSE_STRUCTURED_CREATE_2_4        1992

#define FB_OP_FB_SPARSE_STRUCTURED_GET_COMPRESSION_RATIO 1993

#define FB_OP_FB_SPARSE_SYMM                         1994

#define FB_OP_FB_SPARSE_SYMV                         1995

#define FB_OP_FB_SPARSE_SYR2K                        1996

#define FB_OP_FB_SPARSE_SYRK                         1997

#define FB_OP_FB_SPARSE_TFQMR                        1998

#define FB_OP_FB_SPARSE_TRSM                         1999

#define FB_OP_FB_SPARSE_TRSV                         2000

/* =========================================================================
 * Tensor operations + RNG + NCCL/RCCL collectives  (2001 – 2117)
 * ========================================================================= */

#define FB_OP_FB_BATCH_GEMM_BF16                     2001

#define FB_OP_FB_BATCH_GEMM_FP32                     2002

#define FB_OP_FB_BATCH_GEMM_INT8                     2003

#define FB_OP_FB_BATCH_GEMM_STRIDED_BF16             2004

#define FB_OP_FB_BATCH_GEMM_STRIDED_FP32             2005

#define FB_OP_FB_BATCH_GEMM_STRIDED_INT8             2006

#define FB_OP_FB_CENTER_OF_MASS                      2007

#define FB_OP_FB_CLEBSCH_GORDAN_COEFFICIENTS         2008

#define FB_OP_FB_DBSCAN_FIT                          2009

#define FB_OP_FB_EINSUM                              2010

#define FB_OP_FB_ELASTIC_NET_FIT                     2011

#define FB_OP_FB_EQUIVARIANT_MLP                     2012

#define FB_OP_FB_EQUIVARIANT_POOLING                 2013

#define FB_OP_FB_FEATURE_RECURSIVE_ELIMINATION       2014

#define FB_OP_FB_FEATURE_UNIVARIATE_SELECTION        2015

#define FB_OP_FB_FEATURE_VARIANCE_THRESHOLD          2016

#define FB_OP_FB_GEOMETRIC_MESSAGE_PASSING           2017

#define FB_OP_FB_GROUP_ACTION                        2018

#define FB_OP_FB_HIERARCHICAL_CUT                    2019

#define FB_OP_FB_HIERARCHICAL_FIT                    2020

#define FB_OP_FB_IMPUTE_KNN                          2021

#define FB_OP_FB_IMPUTE_MEAN                         2022

#define FB_OP_FB_IMPUTE_MEDIAN                       2023

#define FB_OP_FB_IRREP_TENSOR_PRODUCT                2024

#define FB_OP_FB_KFOLD_SPLIT                         2025

#define FB_OP_FB_LOGISTIC_REGRESSION_FIT             2026

#define FB_OP_FB_LOGISTIC_REGRESSION_PREDICT         2027

#define FB_OP_FB_LOGISTIC_REGRESSION_PREDICT_PROBA   2028

#define FB_OP_FB_NCCL_ALLGATHER                      2029

#define FB_OP_FB_NCCL_ALLREDUCE                      2030

#define FB_OP_FB_NCCL_ALLTOALL                       2031

#define FB_OP_FB_NCCL_BROADCAST                      2032

#define FB_OP_FB_NCCL_COMM_SPLIT                     2033

#define FB_OP_FB_NCCL_GATHER                         2034

#define FB_OP_FB_NCCL_GROUP_END                      2035

#define FB_OP_FB_NCCL_GROUP_START                    2036

#define FB_OP_FB_NCCL_RECV                           2037

#define FB_OP_FB_NCCL_REDUCE                         2038

#define FB_OP_FB_NCCL_REDUCESCATTER                  2039

#define FB_OP_FB_NCCL_SCATTER                        2040

#define FB_OP_FB_NCCL_SEND                           2041

#define FB_OP_FB_NEIGHBOR_LIST_ALLPAIRS              2042

#define FB_OP_FB_NEIGHBOR_LIST_CELLLIST              2043

#define FB_OP_FB_NEIGHBOR_LIST_VERLET                2044

#define FB_OP_FB_PARITY_TRANSFORM                    2045

#define FB_OP_FB_PME_AUTO_ENERGY                     2046

#define FB_OP_FB_PME_AUTO_FORCES                     2047

#define FB_OP_FB_PME_ENERGY                          2048

#define FB_OP_FB_PME_FORCES                          2049

#define FB_OP_FB_RADIAL_BASIS_FUNCTIONS              2050

#define FB_OP_FB_REDUCE_UNIFIED                      2051

#define FB_OP_FB_REFLECT_POSITIONS                   2052

#define FB_OP_FB_RNG_BERNOULLI                       2053

#define FB_OP_FB_RNG_BETA                            2054

#define FB_OP_FB_RNG_BINOMIAL                        2055

#define FB_OP_FB_RNG_CAUCHY                          2056

#define FB_OP_FB_RNG_CREATE_PHILOX                   2057

#define FB_OP_FB_RNG_DESTROY                         2058

#define FB_OP_FB_RNG_EXPONENTIAL                     2059

#define FB_OP_FB_RNG_GAMMA                           2060

#define FB_OP_FB_RNG_GAUSSIAN                        2061

#define FB_OP_FB_RNG_GET_STATE_SIZE                  2062

#define FB_OP_FB_RNG_LOGNORMAL                       2063

#define FB_OP_FB_RNG_POISSON                         2064

#define FB_OP_FB_RNG_SET_SEED                        2065

#define FB_OP_FB_RNG_SKIP_AHEAD                      2066

#define FB_OP_FB_RNG_UNIFORM                         2067

#define FB_OP_FB_ROTATE_IRREPS                       2068

#define FB_OP_FB_ROTATE_SPHERICAL_HARMONICS          2069

#define FB_OP_FB_SE3_CONVOLUTION                     2070

#define FB_OP_FB_SPHERICAL_HARMONICS                 2071

#define FB_OP_FB_SPHERICAL_TO_CARTESIAN              2072

#define FB_OP_FB_STATS_CENTRAL_MOMENTS               2073

#define FB_OP_FB_STATS_CORRELATION                   2074

#define FB_OP_FB_STATS_CORRELATION_MATRIX            2075

#define FB_OP_FB_STATS_COVARIANCE                    2076

#define FB_OP_FB_STATS_COVARIANCE_MATRIX             2077

#define FB_OP_FB_STATS_HISTOGRAM                     2078

#define FB_OP_FB_STATS_IQR_OUTLIERS                  2079

#define FB_OP_FB_STATS_KDE                           2080

#define FB_OP_FB_STATS_KURTOSIS                      2081

#define FB_OP_FB_STATS_MEAN                          2082

#define FB_OP_FB_STATS_MEDIAN                        2083

#define FB_OP_FB_STATS_MIN_MAX                       2084

#define FB_OP_FB_STATS_MODIFIED_ZSCORE               2085

#define FB_OP_FB_STATS_PERCENTILE                    2086

#define FB_OP_FB_STATS_QUANTILE                      2087

#define FB_OP_FB_STATS_QUARTILES                     2088

#define FB_OP_FB_STATS_RAW_MOMENTS                   2089

#define FB_OP_FB_STATS_SKEWNESS                      2090

#define FB_OP_FB_STATS_STD                           2091

#define FB_OP_FB_STATS_SUM                           2092

#define FB_OP_FB_STATS_VARIANCE                      2093

#define FB_OP_FB_STATS_ZSCORE                        2094

#define FB_OP_FB_SVD_TRUNCATED                       2095

#define FB_OP_FB_TENSOR_ADD                          2096

#define FB_OP_FB_TENSOR_CONTRACT                     2097

#define FB_OP_FB_TENSOR_CONTRACT_BATCHED             2098

#define FB_OP_FB_TENSOR_COPY                         2099

#define FB_OP_FB_TENSOR_GEMM                         2100

#define FB_OP_FB_TENSOR_HADAMARD                     2101

#define FB_OP_FB_TENSOR_MTTKRP                       2102

#define FB_OP_FB_TENSOR_PERMUTE                      2103

#define FB_OP_FB_TENSOR_REDUCE_MAX                   2104

#define FB_OP_FB_TENSOR_REDUCE_MIN                   2105

#define FB_OP_FB_TENSOR_REDUCE_NORM                  2106

#define FB_OP_FB_TENSOR_REDUCE_SUM                   2107

#define FB_OP_FB_TENSOR_RESHAPE                      2108

#define FB_OP_FB_TENSOR_SCALE                        2109

#define FB_OP_FB_TENSOR_TRANSPOSE                    2110

#define FB_OP_FB_TENSOR_TRANSPOSE_SCALE              2111

#define FB_OP_FB_TENSOR_TTM                          2112

#define FB_OP_FB_TENSOR_TTV                          2113

#define FB_OP_FB_TENSOR_VIEW                         2114

#define FB_OP_FB_TRAIN_TEST_SPLIT                    2115

#define FB_OP_FB_TRANSLATE_POSITIONS                 2116

#define FB_OP_FB_WIGNER_D_MATRIX                     2117

/* =========================================================================
 * Statistics + Machine Learning  (2118 – 2141)
 * ========================================================================= */

#define FB_OP_FB_AGGLOMERATIVE_FIT                   2118

#define FB_OP_FB_COVARIANCE_MATRIX_EQUIVARIANT       2119

#define FB_OP_FB_DECISION_TREE_FEATURE_IMPORTANCE    2120

#define FB_OP_FB_DECISION_TREE_FIT                   2121

#define FB_OP_FB_DECISION_TREE_PREDICT               2122

#define FB_OP_FB_KMEANS_FIT                          2123

#define FB_OP_FB_KMEANS_INERTIA                      2124

#define FB_OP_FB_KMEANS_PREDICT                      2125

#define FB_OP_FB_LASSO_REGRESSION_FIT                2126

#define FB_OP_FB_LINEAR_REGRESSION_FIT               2127

#define FB_OP_FB_LINEAR_REGRESSION_PREDICT           2128

#define FB_OP_FB_LINEAR_REGRESSION_SCORE             2129

#define FB_OP_FB_MIN_MAX_SCALE                       2130

#define FB_OP_FB_NORMALIZE                           2131

#define FB_OP_FB_NORMALIZE_UNIFIED                   2132

#define FB_OP_FB_PCA_EXPLAINED_VARIANCE              2133

#define FB_OP_FB_PCA_FIT                             2134

#define FB_OP_FB_PCA_FIT_TRANSFORM                   2135

#define FB_OP_FB_PCA_INVERSE_TRANSFORM               2136

#define FB_OP_FB_PCA_TRANSFORM                       2137

#define FB_OP_FB_PERMUTATION_EQUIVARIANT_ATTENTION   2138

#define FB_OP_FB_PERMUTATION_INVARIANT_AGGREGATION   2139

#define FB_OP_FB_RIDGE_REGRESSION_FIT                2140

#define FB_OP_FB_STANDARDIZE                         2141

/* =========================================================================
 * Parallel Primitives (fb_prim_*)  (2142 – 2176)
 * ========================================================================= */

#define FB_OP_FB_PRIM_ADJACENT_DIFFERENCE            2142

#define FB_OP_FB_PRIM_COPY_IF                        2143

#define FB_OP_FB_PRIM_DISCONTINUITY                  2144

#define FB_OP_FB_PRIM_EXCHANGE_BLOCK_TO_STRIPED      2145

#define FB_OP_FB_PRIM_EXCHANGE_STRIPED_TO_BLOCK      2146

#define FB_OP_FB_PRIM_GATHER                         2147

#define FB_OP_FB_PRIM_HISTOGRAM_EVEN                 2148

#define FB_OP_FB_PRIM_HISTOGRAM_RANGE                2149

#define FB_OP_FB_PRIM_MERGE                          2150

#define FB_OP_FB_PRIM_MERGE_BY_KEY                   2151

#define FB_OP_FB_PRIM_MERGE_SORT                     2152

#define FB_OP_FB_PRIM_NTH_ELEMENT                    2153

#define FB_OP_FB_PRIM_PARTIAL_SORT                   2154

#define FB_OP_FB_PRIM_PARTITION                      2155

#define FB_OP_FB_PRIM_PARTITION_THREE_WAY            2156

#define FB_OP_FB_PRIM_RADIX_SORT                     2157

#define FB_OP_FB_PRIM_REDUCE                         2158

#define FB_OP_FB_PRIM_REDUCE_BLOCK                   2159

#define FB_OP_FB_PRIM_REDUCE_BY_KEY                  2160

#define FB_OP_FB_PRIM_REDUCE_DEVICE                  2161

#define FB_OP_FB_PRIM_REDUCE_WARP                    2162

#define FB_OP_FB_PRIM_SCAN_BY_KEY                    2163

#define FB_OP_FB_PRIM_SCAN_EXCLUSIVE                 2164

#define FB_OP_FB_PRIM_SCAN_INCLUSIVE                 2165

#define FB_OP_FB_PRIM_SCATTER                        2166

#define FB_OP_FB_PRIM_SELECT_FLAGGED                 2167

#define FB_OP_FB_PRIM_SELECT_IF                      2168

#define FB_OP_FB_PRIM_SHUFFLE_ROTATE                 2169

#define FB_OP_FB_PRIM_SHUFFLE_UP                     2170

#define FB_OP_FB_PRIM_SORT_KEYS                      2171

#define FB_OP_FB_PRIM_SORT_PAIRS                     2172

#define FB_OP_FB_PRIM_TRANSFORM                      2173

#define FB_OP_FB_PRIM_TRANSFORM_IF                   2174

#define FB_OP_FB_PRIM_UNIQUE                         2175

#define FB_OP_FB_PRIM_UNIQUE_BY_KEY                  2176

/* =========================================================================
 * Chemistry / Physics extensions  (2177 – 2184)
 * ========================================================================= */

#define FB_OP_FB_COULOMB_DIRECT_ENERGY               2177

#define FB_OP_FB_COULOMB_DIRECT_FORCES               2178

#define FB_OP_FB_DFTD3_DISPERSION_ENERGY             2179

#define FB_OP_FB_DFTD3_DISPERSION_FORCES             2180

#define FB_OP_FB_DFTD4_DISPERSION_ENERGY             2181

#define FB_OP_FB_DFTD4_DISPERSION_FORCES             2182

#define FB_OP_FB_EWALD_ENERGY                        2183

#define FB_OP_FB_EWALD_FORCES                        2184

/* =========================================================================
 * Vector math (fb_v*)  (2185 – 2217)
 * ========================================================================= */

#define FB_OP_FB_VACOS                               2185

#define FB_OP_FB_VACOSH                              2186

#define FB_OP_FB_VASIN                               2187

#define FB_OP_FB_VASINH                              2188

#define FB_OP_FB_VATAN                               2189

#define FB_OP_FB_VATANH                              2190

#define FB_OP_FB_VCBRT                               2191

#define FB_OP_FB_VCOS                                2192

#define FB_OP_FB_VCOSH                               2193

#define FB_OP_FB_VERF                                2194

#define FB_OP_FB_VERFC                               2195

#define FB_OP_FB_VERFINV                             2196

#define FB_OP_FB_VEXP                                2197

#define FB_OP_FB_VEXP2                               2198

#define FB_OP_FB_VGAMMA                              2199

#define FB_OP_FB_VINVCBRT                            2200

#define FB_OP_FB_VINVSQRT                            2201

#define FB_OP_FB_VJ0                                 2202

#define FB_OP_FB_VJ1                                 2203

#define FB_OP_FB_VLGAMMA                             2204

#define FB_OP_FB_VLOG                                2205

#define FB_OP_FB_VLOG10                              2206

#define FB_OP_FB_VLOG2                               2207

#define FB_OP_FB_VPOW                                2208

#define FB_OP_FB_VPOW2O3                             2209

#define FB_OP_FB_VPOW3O2                             2210

#define FB_OP_FB_VSIN                                2211

#define FB_OP_FB_VSINH                               2212

#define FB_OP_FB_VSQRT                               2213

#define FB_OP_FB_VTAN                                2214

#define FB_OP_FB_VTANH                               2215

#define FB_OP_FB_VY0                                 2216

#define FB_OP_FB_VY1                                 2217

/* =========================================================================
 * Spline interpolation (fb_spline_*)  (2218 – 2233)
 * ========================================================================= */

#define FB_OP_FB_SPLINE_CREATE_1D_AKIMA              2218

#define FB_OP_FB_SPLINE_CREATE_1D_BESSEL             2219

#define FB_OP_FB_SPLINE_CREATE_1D_CUBIC              2220

#define FB_OP_FB_SPLINE_CREATE_1D_HERMITE            2221

#define FB_OP_FB_SPLINE_CREATE_1D_LINEAR             2222

#define FB_OP_FB_SPLINE_CREATE_1D_QUADRATIC          2223

#define FB_OP_FB_SPLINE_CREATE_2D_CUBIC              2224

#define FB_OP_FB_SPLINE_CREATE_2D_LINEAR             2225

#define FB_OP_FB_SPLINE_DESTROY                      2226

#define FB_OP_FB_SPLINE_EVAL                         2227

#define FB_OP_FB_SPLINE_EVAL_BATCH                   2228

#define FB_OP_FB_SPLINE_EVAL_DERIVATIVE              2229

#define FB_OP_FB_SPLINE_EVAL_INTEGRAL                2230

#define FB_OP_FB_SPLINE_GET_BREAKPOINTS              2231

#define FB_OP_FB_SPLINE_GET_COEFFICIENTS             2232

#define FB_OP_FB_SPLINE_SEARCH_CELL                  2233

/* =========================================================================
 * GEMM / LPGEMM fused extensions  (2234 – 2265)
 * ========================================================================= */

#define FB_OP_FB_GEMM_BIAS_BF16                      2234

#define FB_OP_FB_GEMM_BIAS_FP32                      2235

#define FB_OP_FB_GEMM_BIAS_GELU_BF16                 2236

#define FB_OP_FB_GEMM_BIAS_GELU_FP32                 2237

#define FB_OP_FB_GEMM_BIAS_INT8                      2238

#define FB_OP_FB_GEMM_BIAS_LAYERNORM_BF16            2239

#define FB_OP_FB_GEMM_BIAS_LAYERNORM_FP32            2240

#define FB_OP_FB_GEMM_BIAS_RELU_BF16                 2241

#define FB_OP_FB_GEMM_BIAS_RELU_FP32                 2242

#define FB_OP_FB_GEMM_BIAS_RESIDUAL_RELU_BF16        2243

#define FB_OP_FB_GEMM_BIAS_RESIDUAL_RELU_FP32        2244

#define FB_OP_FB_GEMM_BIAS_SIGMOID_BF16              2245

#define FB_OP_FB_GEMM_BIAS_SIGMOID_FP32              2246

#define FB_OP_FB_GEMM_BIAS_TANH_BF16                 2247

#define FB_OP_FB_GEMM_BIAS_TANH_FP32                 2248

#define FB_OP_FB_GEMM_FP8_E4M3                       2249

#define FB_OP_FB_GEMM_FUSED_BIAS_ADD                 2250

#define FB_OP_FB_GEMM_FUSED_GELU                     2251

#define FB_OP_FB_GEMM_FUSED_RELU                     2252

#define FB_OP_FB_GEMM_FUSED_SCALE_ACT                2253

#define FB_OP_FB_GEMM_FUSED_SILU                     2254

#define FB_OP_FB_GEMM_FUSED_SWISH                    2255

#define FB_OP_FB_GEMM_INT4                           2256

#define FB_OP_FB_GEMM_UNIFIED                        2257

#define FB_OP_FB_LPGEMM_BF16                         2258

#define FB_OP_FB_LPGEMM_BF16_BF16                    2259

#define FB_OP_FB_LPGEMM_FP32                         2260

#define FB_OP_FB_LPGEMM_INT8                         2261

#define FB_OP_FB_LPGEMM_INT8_FP32                    2262

#define FB_OP_FB_QGEMM_BIAS_GELU_INT8                2263

#define FB_OP_FB_QGEMM_BIAS_RELU_INT8                2264

#define FB_OP_FB_QGEMM_INT8_ASYMMETRIC               2265

#endif /* FB_JUDGE_OP_IDS_H */
