#!/usr/bin/env python3
"""
Full regeneration of judge_op_ids.h:
  - Family-first ordering (S/D/C/Z consecutive) within every section
  - Dense IDs from 0, no gaps anywhere
  - Section boundary sentinels added (FB_OP__BLAS_L1_END, etc.)
  - All sections covered; header comment updated with new ranges
"""

import re
from collections import defaultdict, OrderedDict

SRC = 'src/judge/judge_op_ids.h'

# ── 1. Read existing symbols ─────────────────────────────────────────────────
lines = open(SRC, encoding='utf-8').readlines()
sym_order = []   # preserve order for extraction
sym_to_id = {}
for l in lines:
    m = re.match(r'#define (FB_OP_\w+)\s+(\d+)', l.strip())
    if m:
        sym = m.group(1)
        if sym.startswith('FB_OP__'):   # skip sentinel defines (double-underscore)
            continue
        sym_order.append(sym)
        sym_to_id[sym] = int(m.group(2))

print(f'Read {len(sym_order)} symbols')

# ── 2. Known BLAS operation sets (for section classification) ────────────────

# Explicit op-name whitelists for L1/L2/L3 — more reliable than base-stripping
# because some L1 ops have irregular names (CSSCAL, ISAMAX, SDSDOT, etc.)
BLAS_L1_NAMES = {
    'SAXPY','DAXPY','CAXPY','ZAXPY',
    'SSCAL','DSCAL','CSCAL','ZSCAL','CSSCAL','ZDSCAL',
    'SCOPY','DCOPY','CCOPY','ZCOPY',
    'SSWAP','DSWAP','CSWAP','ZSWAP',
    'SDOT','DDOT','CDOTC','CDOTU','ZDOTC','ZDOTU',
    'SDSDOT','DSDOT',
    'SASUM','DASUM','SCASUM','DZASUM',
    'SNRM2','DNRM2','SCNRM2','DZNRM2',
    'ISAMAX','IDAMAX','ICAMAX','IZAMAX',
    'SROT','DROT','CROT','ZROT','ZDROT',
    'SROTG','DROTG',
    'SROTM','DROTM',
    'SROTMG','DROTMG',
}
BLAS_L2_NAMES = {
    'SGEMV','DGEMV','CGEMV','ZGEMV',
    'SSYMV','DSYMV','CHEMV','ZHEMV',
    'STRMV','DTRMV','CTRMV','ZTRMV',
    'STRSV','DTRSV','CTRSV','ZTRSV',
    'SGER','DGER','CGERU','CGERC','ZGERU','ZGERC',
    'SSYR','DSYR','CHER','ZHER',
    'SSYR2','DSYR2','CHER2','ZHER2',
    'SSPMV','DSPMV','CHPMV','ZHPMV',
    'SSBMV','DSBMV','CHBMV','ZHBMV',
    'STBMV','DTBMV','CTBMV','ZTBMV',
    'STBSV','DTBSV','CTBSV','ZTBSV',
    'STPMV','DTPMV','CTPMV','ZTPMV',
    'STPSV','DTPSV','CTPSV','ZTPSV',
    'SSPR','DSPR','CHPR','ZHPR',
    'SSPR2','DSPR2','CHPR2','ZHPR2',
    'SGBMV','DGBMV','CGBMV','ZGBMV',
    'SGBMVX','DGBMVX','CGBMVX','ZGBMVX',
}
BLAS_L3_NAMES = {
    'SGEMM','DGEMM','CGEMM','ZGEMM',
    'SSYMM','DSYMM','CSYMM','ZSYMM',
    'CHEMM','ZHEMM',
    'SSYRK','DSYRK','CSYRK','ZSYRK',
    'CHERK','ZHERK',
    'SSYR2K','DSYR2K','CSYR2K','ZSYR2K',
    'CHER2K','ZHER2K',
    'STRMM','DTRMM','CTRMM','ZTRMM',
    'STRSM','DTRSM','CTRSM','ZTRSM',
    'SGEMM_BATCH','DGEMM_BATCH','CGEMM_BATCH','ZGEMM_BATCH',
    'SGEMM_STRIDED','DGEMM_STRIDED','CGEMM_STRIDED','ZGEMM_STRIDED',
    'CGEMM3M','ZGEMM3M',
}

# Keep base-name sets for the fallback LAPACK classifier
L2_BASE = {
    'GEMV','SYMV','HEMV','TRMV','TRSV','GER','GERU','GERC',
    'SYR','HER','SYR2','HER2','SPMV','HPMV','SBMV','HBMV',
    'TBMV','TBSV','TPMV','TPSV','SPR','HPR','SPR2','HPR2',
    'GBMV','GBMVX',
}
L3_BASE = {
    'GEMM','SYMM','HEMM','SYRK','HERK','SYR2K','HER2K','TRMM','TRSM',
    'GEMM3M',
}

# Family key overrides for irregular L1 ops (so they sort within a consistent family)
FAMILY_OVERRIDE = {
    # mixed-precision scale: CSSCAL and ZDSCAL group with S/D/C/ZSCAL family
    'FB_OP_CSSCAL': ('SCAL', 2),   # C-precision
    'FB_OP_ZDSCAL': ('SCAL', 3),   # Z-precision
    # mixed-return-type ASUM / NRM2
    'FB_OP_SCASUM': ('ASUM', 2),   # C-magnitude, S-return
    'FB_OP_DZASUM': ('ASUM', 3),   # Z-magnitude, D-return
    'FB_OP_SCNRM2': ('NRM2', 2),   # C-input, S-return
    'FB_OP_DZNRM2': ('NRM2', 3),   # Z-input, D-return
    # mixed-precision dot: SDSDOT (returns D, inputs S/S) and DSDOT (D from S)
    'FB_OP_SDSDOT': ('DSDOT', 0),  # treat as S-variant of DSDOT family
    'FB_OP_DSDOT':  ('DSDOT', 1),  # treat as D-variant
    # index-return AMAX family: ISAMAX(S), IDAMAX(D), ICAMAX(C), IZAMAX(Z)
    'FB_OP_ISAMAX': ('IAMAX', 0),
    'FB_OP_IDAMAX': ('IAMAX', 1),
    'FB_OP_ICAMAX': ('IAMAX', 2),
    'FB_OP_IZAMAX': ('IAMAX', 3),
}
L3_BATCH_SUFFIXES = ('_BATCH', '_STRIDED', '_BATCH_STRIDED')

SCALAPACK_PREFIXES = (
    'FB_OP_PS', 'FB_OP_PD', 'FB_OP_PC', 'FB_OP_PZ',
)

KNOWN_LAPACK_BASES = {
    # LU / Cholesky / QR / LDL factorizations
    'GETRF','GETRS','GETRI','GETF2','GBTRF','GBTRS',
    'POTRF','POTRS','POTRI','PBTRF','PBTRS','PTTRF','PTTRS',
    'SYTRF','HETRF','SYTRS','HETRS','SYTRI','HETRI',
    'GEQRF','GEQP3','GEQPF','ORMQR','UNMQR','ORGRQ','UNGRQ',
    'GELQF','ORMLQ','UNMLQ','ORGQR','UNGQR',
    # Solves
    'GESV','GESVX','GBSV','GBSVX','POSV','POSVX','PBSV','PBSVX',
    'PTSV','PTSVX','SYSV','HESV','SYSVX','HESVX','GTSV','GTSVX',
    'GELS','GELSD','GELSS','GELSY',
    # Eigenvalue
    'SYEV','HEEV','SYEVD','HEEVD','SYEVR','HEEVR','SYEVX','HEEVX',
    'GEEV','GEEVX','GGEV','GGEVX','GGEVD',
    'SBEV','HBEV','SBEVD','HBEVD','SBEVX','HBEVX',
    'STEV','STEBZ','STEGR','STEIN','STERF','STEDC','STEQR',
    'SPEV','HPEV','SPEVD','HPEVD','SPEVX','HPEVX',
    # SVD
    'GESVD','GESDD','GESVDX','GESVDQ','BDSVDX','BDSQR','BDSDC',
    'GGSVD','GGSVD3','GGSVP','GGSVP3',
    # Least squares / conditioning
    'GELSD','GELSS','GELSY','GELS',
    'GECON','GBCON','POCON','PBCON','PTCON','SYCON','HECON','TRCON',
    'GEERX','GERFS','GBRFS','PORFS','PBRFS','PTRFS','SYRFS','HERFS','TRRFS',
    'GEBAL','GEBAK','GEHRD','ORGHR','UNGHR','ORMHR','UNMHR',
    'HSEQR','TREVC','TREXC','TRSNA','TRSYL','TGEXC','TGSEN','TGSNA','TGSYL',
    'GGES','GGESX','GGEV','GGEVX','GGGLM','GGHRD','GGHD3','GGQRF','GGRQF',
    'GGLSE','GGSVD3','GGSVP3',
    # Auxiliary / miscellaneous LAPACK routines
    'LACPY','LASET','LASWP','LASYF','LAHEF','LATRD','LATRS',
    'LAUUM','LAUU2','LARGE','LARF','LARFB','LARFG','LARFGP','LARFT','LARFX','LARFY',
    'LARNV','LAGTM','LAGGE','LAGSY','LAGHE','LAGTR','LAGTF','LAGTS',
    'LAPY2','LAPY3','LAPLL','LAPMT','LARTG','LARTGP','LARTZ',
    'LACON','LACN2','LALS0','LALSA','LALSD','LASDT','LASD0','LASD1',
    'LASD2','LASD3','LASD4','LASD5','LASD6','LASD7','LASD8','LASDA',
    'LASDQ','LASDT','LASQ1','LASQ2','LASQ3','LASQ4','LASQ5','LASQ6',
    'LASR','LASRT','LASSQ','LASTC','LAGTQ','LATBS','LATDF','LATPS',
    'LAUU2','LAUUM',
    # LWORK query
    'ILAENV','ILAPREC',
    # Banded / tridiagonal LAPACK
    'GBSV','GBTRS','GBTRF',
    # Other solver routines
    'PTSVX','SISNAN','SLAISNAN','DLAISNAN',
    # Mixed-precision
    'DSGESV','ZCGESV','DSPOSV','ZCPOSV',
    # Extra
    'GESVDA','GESVDP',
}


def classify(sym):
    """Return section name for symbol."""
    name = sym[len('FB_OP_'):]

    # Explicit L1/L2/L3 whitelists — checked first (overrides pattern logic)
    if name in BLAS_L1_NAMES:   return 'BLAS_L1'
    if name in BLAS_L2_NAMES:   return 'BLAS_L2'
    if name in BLAS_L3_NAMES:   return 'BLAS_L3'

    # CBLAS_* → Extended BLAS
    if name.startswith('CBLAS_'):
        return 'EXTENDED_BLAS'

    # MKL_* → MKL
    if name.startswith('MKL_'):
        return 'MKL'

    # FB_ prefix → specialty sections
    if name.startswith('FB_'):
        sub = name[3:]  # strip FB_
        if sub.startswith('DNN_'):      return 'DNN'
        if sub.startswith('FFT_'):      return 'FFT'
        if sub.startswith('SPARSE_'):   return 'SPARSE'
        if sub.startswith('PRIM_'):     return 'PARALLEL_PRIMS'
        if sub.startswith('SPLINE_'):   return 'SPLINE'
        if sub.startswith('GEMM_') or sub.startswith('LPGEMM_') or sub.startswith('QGEMM_'):
            return 'GEMM_FUSED'
        if sub.startswith('RNG_') or sub.startswith('TENSOR_') or \
           sub.startswith('NCCL_') or sub.startswith('RCCL_') or \
           sub.startswith('ALLREDUCE') or sub.startswith('REDUCE') or \
           sub.startswith('BROADCAST') or sub.startswith('ALLGATHER') or \
           sub.startswith('SCATTER') or sub.startswith('BARRIER'):
            return 'TENSOR_RNG_NCCL'
        if sub.startswith('NORMALIZE') or sub.startswith('STANDARDIZE') or \
           sub.startswith('MEAN') or sub.startswith('VAR') or \
           sub.startswith('COVAR') or sub.startswith('CORR') or \
           sub.startswith('HISTOGRAM') or sub.startswith('PERCENTILE') or \
           sub.startswith('ZSCORE') or sub.startswith('MIN_MAX') or \
           sub.startswith('KMEANS') or sub.startswith('PCA') or \
           sub.startswith('LINEAR_REGRESSION') or sub.startswith('RIDGE') or \
           sub.startswith('LASSO') or sub.startswith('SVR') or \
           sub.startswith('RANDOM_FOREST') or sub.startswith('GRADIENT') or \
           sub.startswith('NEURAL') or sub.startswith('AGGLOMERAT') or \
           sub.startswith('DECISION_TREE') or sub.startswith('PERMUTATION'):
            return 'STATS_ML'
        if sub.startswith('HARTREE') or sub.startswith('FOCK') or \
           sub.startswith('COULOMB') or sub.startswith('DFT') or \
           sub.startswith('EXCHANGE') or sub.startswith('CORRELATION') or \
           sub.startswith('INTEG') or sub.startswith('EWALD') or \
           sub.startswith('PPPM') or sub.startswith('PAIR') or \
           sub.startswith('BOND') or sub.startswith('ANGLE') or \
           sub.startswith('DIHEDRAL') or sub.startswith('TORSION') or \
           sub.startswith('VDW') or sub.startswith('LENNARD'):
            return 'CHEM_PHYS'
        if sub.startswith('V') and len(sub) > 1 and sub[1].isupper():
            return 'VECTOR_MATH'
        return 'TENSOR_RNG_NCCL'

    # Batch/strided suffix → Extended BLAS
    if any(name.endswith(s) for s in ('_BATCH_STRIDED', '_BATCH', '_STRIDED')):
        return 'EXTENDED_BLAS'

    # AXPBY variants → Extended BLAS
    if name.endswith('AXPBY'):
        return 'EXTENDED_BLAS'

    # ScaLAPACK: P + SDCZ + known base
    if len(name) >= 3 and name[0] == 'P' and name[1] in 'SDCZ':
        return 'SCALAPACK'

    # Mixed-precision LAPACK drivers
    if name in ('DSGESV','ZCGESV','DSPOSV','ZCPOSV'):
        return 'LAPACK'

    # Standard LAPACK: remaining S/D/C/Z-prefixed ops
    if len(name) >= 2 and name[0] in 'SDCZ':
        return 'LAPACK'

    # ILAENV and similar utility functions
    if name.startswith('ILAENV') or name.startswith('ILAPREC'):
        return 'LAPACK'

    return 'OTHER'

# ── 3. Classify every symbol ─────────────────────────────────────────────────
section_syms = defaultdict(list)
for sym in sym_order:
    section_syms[classify(sym)].append(sym)

# Report classification
SECTION_ORDER = [
    'BLAS_L1', 'BLAS_L2', 'BLAS_L3',
    'LAPACK', 'SCALAPACK',
    'EXTENDED_BLAS', 'MKL',
    'DNN', 'FFT', 'SPARSE',
    'TENSOR_RNG_NCCL', 'STATS_ML',
    'PARALLEL_PRIMS', 'CHEM_PHYS',
    'VECTOR_MATH', 'SPLINE',
    'GEMM_FUSED', 'OTHER',
]
for sec in SECTION_ORDER:
    print(f'  {sec:22}: {len(section_syms[sec])} ops')

# Verify all symbols classified
all_classified = sum(len(v) for v in section_syms.values())
assert all_classified == len(sym_order), f"Mismatch: {all_classified} vs {len(sym_order)}"

# ── 4. Family-first sort within each section ─────────────────────────────────

PREC_ORDER = {'S': 0, 'D': 1, 'C': 2, 'Z': 3, 'I': 4}

def precision_and_base(sym):
    name = sym[len('FB_OP_'):]
    if len(name) >= 2 and name[0] in 'SDCZI':
        prec = name[0]
        base = name[1:]
        if len(base) >= 2:
            return prec, base
    return None, name

def family_sort_key(sym):
    if sym in FAMILY_OVERRIDE:
        base, prec_ord = FAMILY_OVERRIDE[sym]
        return (base, prec_ord, '')
    prec, base = precision_and_base(sym)
    if prec is None:
        # Non-precision op: sort by full name after FB_OP_
        return (sym[len('FB_OP_'):], 5, '')
    return (base, PREC_ORDER.get(prec, 9), prec)

def sym_prec_order(sym):
    """Sort key for precision ordering within a family, respecting FAMILY_OVERRIDE."""
    if sym in FAMILY_OVERRIDE:
        return FAMILY_OVERRIDE[sym][1]
    prec, _ = precision_and_base(sym)
    return PREC_ORDER.get(prec or 'X', 5)

def sym_base(sym):
    """Get the family base name for grouping, respecting FAMILY_OVERRIDE."""
    if sym in FAMILY_OVERRIDE:
        return FAMILY_OVERRIDE[sym][0]
    _, base = precision_and_base(sym)
    return base

sorted_section_syms = {}
for sec in SECTION_ORDER:
    syms = section_syms[sec]
    # Group by base, sort groups alphabetically, sort within group by precision
    family = defaultdict(list)
    for sym in syms:
        family[sym_base(sym)].append(sym)

    result = []
    for base in sorted(family.keys()):
        members = sorted(family[base], key=sym_prec_order)
        result.extend(members)
    sorted_section_syms[sec] = result

# ── 5. Assign dense IDs from 0 ───────────────────────────────────────────────

new_id_of = {}
section_ranges = {}   # sec -> (first_id, last_id)
next_id = 0

for sec in SECTION_ORDER:
    syms = sorted_section_syms[sec]
    if not syms:
        section_ranges[sec] = None
        continue
    first = next_id
    for sym in syms:
        new_id_of[sym] = next_id
        next_id += 1
    section_ranges[sec] = (first, next_id - 1)

total_ops = next_id
print(f'\nNew ID range: 0 – {total_ops - 1}  ({total_ops} total)')
for sec, r in section_ranges.items():
    if r:
        print(f'  {sec:22}: {r[0]} – {r[1]}')

# FB_JUDGE_MAX_OPERATIONS: round up to next multiple of 100 with some headroom
FB_MAX = ((total_ops + 200) // 100) * 100
print(f'\nFB_JUDGE_MAX_OPERATIONS = {FB_MAX}')

# ── 6. Emit section-boundary sentinels ───────────────────────────────────────
# sentinel: FB_OP__<SECTION>_BEGIN and _END (exclusive upper bound)
# These let corpus.c do range checks without hardcoded numbers.

SENTINEL_NAMES = {
    'BLAS_L1':         ('BLAS_L1_BEGIN', 'BLAS_L1_END'),
    'BLAS_L2':         ('BLAS_L2_BEGIN', 'BLAS_L2_END'),
    'BLAS_L3':         ('BLAS_L3_BEGIN', 'BLAS_L3_END'),
    'LAPACK':          ('LAPACK_BEGIN',   'LAPACK_END'),
    'SCALAPACK':       ('SCALAPACK_BEGIN','SCALAPACK_END'),
}

sentinels = {}  # name -> value
for sec, (bname, ename) in SENTINEL_NAMES.items():
    r = section_ranges.get(sec)
    if r:
        sentinels[f'FB_OP__{bname}'] = r[0]
        sentinels[f'FB_OP__{ename}'] = r[1] + 1  # exclusive

# ── 7. Build section label strings for output ────────────────────────────────

SECTION_LABEL = {
    'BLAS_L1':        'BLAS Level 1',
    'BLAS_L2':        'BLAS Level 2',
    'BLAS_L3':        'BLAS Level 3 (core + batched GEMM)',
    'LAPACK':         'LAPACK — standard s/d/c/z routines',
    'SCALAPACK':      'ScaLAPACK — parallel p* distributed routines',
    'EXTENDED_BLAS':  'Extended BLAS (cblas_batch / cblas_strided / axpby variants)',
    'MKL':            'MKL extensions (mkl_jit, mkl_*omatcopy, mkl_sparse_*)',
    'DNN':            'Deep Neural Network primitives (fb_dnn_*)',
    'FFT':            'FFT (fb_fft_*)',
    'SPARSE':         'Sparse linear algebra (fb_sparse_*)',
    'TENSOR_RNG_NCCL':'Tensor operations + RNG + NCCL/RCCL collectives',
    'STATS_ML':       'Statistics + Machine Learning',
    'PARALLEL_PRIMS': 'Parallel Primitives (fb_prim_*)',
    'CHEM_PHYS':      'Chemistry / Physics extensions',
    'VECTOR_MATH':    'Vector math (fb_v*)',
    'SPLINE':         'Spline interpolation (fb_spline_*)',
    'GEMM_FUSED':     'GEMM / LPGEMM fused extensions',
    'OTHER':          'Other fb_* operations',
}

# ── 8. Build file header comment ─────────────────────────────────────────────

def range_str(sec):
    r = section_ranges.get(sec)
    if not r:
        return '(empty)'
    return f'{r[0]} \u2013 {r[1]}'

block_lines = []
block_lines.append(f' * Block assignments (family-first ordering, S\u2192D\u2192C\u2192Z within each family):')
for sec in SECTION_ORDER:
    r = section_ranges.get(sec)
    if r:
        label = SECTION_LABEL[sec]
        block_lines.append(f' *   {r[0]:5} \u2013 {r[1]:5} : {label}')
block_lines.append(f' *')
block_lines.append(f' * FB_JUDGE_MAX_OPERATIONS: {FB_MAX}')
block_lines.append(f' *')
block_lines.append(f' * Total concrete operation names covered: {total_ops}')
block_lines.append(f' * (see FASTER-BLASTER-OPERATIONS-LIST-APPENDIX.md)')

# ── 9. Emit the new file ─────────────────────────────────────────────────────

out = []
out.append('/**')
out.append(' * @file judge_op_ids.h')
out.append(' * @brief Canonical operation ID assignments for the judge metadata table.')
out.append(' *')
out.append(' * Op IDs are globally unique integers that index into fb_op_judge_table[]')
out.append(' * and fb_benchmark_stats_t arrays.  Every operation in the faster-blaster')
out.append(' * operation superset has exactly one ID here.  Archetype-specific source')
out.append(' * files consume IDs from the ranges assigned below; they do not define')
out.append(' * their own ID namespaces.')
out.append(' *')
out.append(' * An ID being defined here does NOT imply a judge implementation exists yet.')
out.append(' * Unimplemented slots return FB_JUDGE_NOT_IMPLEMENTED at runtime.')
out.append(' *')
for bl in block_lines:
    out.append(bl)
out.append(' *')
out.append(' * Use the FB_OP__*_BEGIN / _END sentinels (not raw numbers) wherever')
out.append(' * section boundary checks are needed in C source.')
out.append(' *')
out.append(' * @copyright Copyright (c) 2025')
out.append(' * @license MIT OR Apache-2.0')
out.append(' */')
out.append('')
out.append('#ifndef FB_JUDGE_OP_IDS_H')
out.append('#define FB_JUDGE_OP_IDS_H')
out.append('')
out.append(f'#define FB_JUDGE_MAX_OPERATIONS {FB_MAX}')
out.append('')
out.append('/* Section boundary sentinels — use these instead of raw numbers. */')
for sname, sval in sorted(sentinels.items(), key=lambda x: x[1]):
    out.append(f'#define {sname:<44} {sval}')
out.append('')

# Emit each section's defines
for sec in SECTION_ORDER:
    syms = sorted_section_syms[sec]
    if not syms:
        continue
    r = section_ranges[sec]
    label = SECTION_LABEL[sec]
    out.append('/* ' + '=' * 73)
    out.append(f' * {label}  ({r[0]} \u2013 {r[1]})')
    out.append(' * ' + '=' * 73 + ' */')
    out.append('')

    # Emit defines, blank line between families
    prev_base = None
    for sym in syms:
        _, base = precision_and_base(sym)
        if prev_base is not None and base != prev_base:
            out.append('')
        out.append(f'#define {sym:<44} {new_id_of[sym]}')
        prev_base = base

    out.append('')

out.append('#endif /* FB_JUDGE_OP_IDS_H */')

content = '\n'.join(out) + '\n'

with open(SRC, 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print(f'\nWritten {SRC}  ({len(content.splitlines())} lines)')

# ── 10. Emit corpus.c patch info ─────────────────────────────────────────────
l1_end = sentinels.get('FB_OP__BLAS_L1_END', '???')
l2_end = sentinels.get('FB_OP__BLAS_L2_END', '???')
l3_end = sentinels.get('FB_OP__BLAS_L3_END', '???')
l1_beg = sentinels.get('FB_OP__BLAS_L1_BEGIN', 0)
l2_beg = sentinels.get('FB_OP__BLAS_L2_BEGIN', '???')
l3_beg = sentinels.get('FB_OP__BLAS_L3_BEGIN', '???')
print(f'\njudge_corpus.c patch: replace hardcoded 48u / 130u:')
print(f'  op_id < 48u   -> op_id < FB_OP__BLAS_L2_BEGIN  (= {l2_beg})')
print(f'  op_id < 130u  -> op_id < FB_OP__BLAS_L3_BEGIN  (= {l3_beg})')
