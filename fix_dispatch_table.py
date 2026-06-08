#!/usr/bin/env python3
"""Convert judge_direct.c dispatch table from hardcoded integers to symbolic FB_OP_ names."""

import re

fname = 'src/judge/judge_direct.c'
lines = open(fname, encoding='utf-8').readlines()

# Find the dispatch table start/end
table_start = None  # line index of the opening brace line
table_end = None    # line index of the closing '};' line
for i, l in enumerate(lines):
    if 'static const fb_direct_runner_fn fb_direct_dispatch' in l:
        table_start = i + 1  # line after the opening {
    if table_start and i > table_start and l.strip() == '};':
        table_end = i
        break

print(f"Table: lines {table_start+1}–{table_end+1} (1-based)")

# Parse each entry in the table
# Patterns:
#   /* N */ runner,  /* FB_OP_XXX */
#   /* N */ runner,  /* FB_OP_XXX — comment */
#   [N] = runner,    /* FB_OP_XXX */
#   [N] = NULL, [M] = NULL, ... (filler lines)
#   /* L3 */

entries = {}  # op_symbol -> runner_expr (e.g. 'run_saxpy' or 'NULL')
not_impl = set()  # symbols that are NULL + "not yet implemented"

for i in range(table_start, table_end):
    l = lines[i]
    stripped = l.strip()
    if not stripped or stripped.startswith('/*') and 'FB_OP_' not in stripped:
        continue

    # positional: /* N */ runner,  /* FB_OP_XXX ... */
    m = re.match(r'/\*\s*\d+\s*\*/\s*(\w+),.*?/\*\s*(FB_OP_\w+)', stripped)
    if m:
        runner, sym = m.group(1), m.group(2)
        entries[sym] = runner
        if runner == 'NULL' and 'not yet' in stripped.lower():
            not_impl.add(sym)
        continue

    # designated: [N] = runner,  /* FB_OP_XXX ... */
    # may have multiple on one line separated by commas
    for part in re.split(r',\s*(?=\[)', stripped):
        m2 = re.match(r'\[(\d+)\]\s*=\s*(\w+).*?/\*\s*(FB_OP_\w+)', part)
        if m2:
            runner, sym = m2.group(2), m2.group(3)
            entries[sym] = runner
            if runner == 'NULL' and 'not yet' in part.lower():
                not_impl.add(sym)
        # NULL fillers without comment: [N] = NULL — skip, C will zero-init
        # (these are [110]=NULL through [129]=NULL etc.)

print(f"Parsed {len(entries)} entries")

# Define canonical order (mirrors judge_op_ids.h block order)
order = [
    # L1
    'FB_OP_SAXPY','FB_OP_DAXPY','FB_OP_CAXPY','FB_OP_ZAXPY',
    'FB_OP_SSCAL','FB_OP_DSCAL','FB_OP_CSCAL','FB_OP_ZSCAL','FB_OP_CSSCAL','FB_OP_ZDSCAL',
    'FB_OP_SCOPY','FB_OP_DCOPY','FB_OP_CCOPY','FB_OP_ZCOPY',
    'FB_OP_SSWAP','FB_OP_DSWAP','FB_OP_CSWAP','FB_OP_ZSWAP',
    'FB_OP_SDOT','FB_OP_DDOT','FB_OP_CDOTC','FB_OP_CDOTU','FB_OP_ZDOTC','FB_OP_ZDOTU',
    'FB_OP_SDSDOT','FB_OP_DSDOT',
    'FB_OP_SASUM','FB_OP_DASUM','FB_OP_SCASUM','FB_OP_DZASUM',
    'FB_OP_SNRM2','FB_OP_DNRM2','FB_OP_SCNRM2','FB_OP_DZNRM2',
    'FB_OP_ISAMAX','FB_OP_IDAMAX','FB_OP_ICAMAX','FB_OP_IZAMAX',
    'FB_OP_SROT','FB_OP_DROT','FB_OP_CROT','FB_OP_ZROT','FB_OP_ZDROT',
    'FB_OP_SROTG','FB_OP_DROTG','FB_OP_SROTM','FB_OP_DROTM','FB_OP_SROTMG',
    # L2
    'FB_OP_SGEMV','FB_OP_DGEMV','FB_OP_CGEMV','FB_OP_ZGEMV',
    'FB_OP_SSYMV','FB_OP_DSYMV','FB_OP_CHEMV','FB_OP_ZHEMV',
    'FB_OP_STRMV','FB_OP_DTRMV','FB_OP_CTRMV','FB_OP_ZTRMV',
    'FB_OP_STRSV','FB_OP_DTRSV','FB_OP_CTRSV','FB_OP_ZTRSV',
    'FB_OP_SGER','FB_OP_DGER','FB_OP_CGERU','FB_OP_CGERC','FB_OP_ZGERU','FB_OP_ZGERC',
    'FB_OP_SSYR','FB_OP_DSYR','FB_OP_CHER','FB_OP_ZHER',
    'FB_OP_SSYR2','FB_OP_DSYR2','FB_OP_CHER2','FB_OP_ZHER2',
    'FB_OP_SSPMV','FB_OP_DSPMV','FB_OP_CHPMV','FB_OP_ZHPMV',
    'FB_OP_SSBMV','FB_OP_DSBMV','FB_OP_CHBMV','FB_OP_ZHBMV',
    'FB_OP_STBMV','FB_OP_DTBMV','FB_OP_CTBMV','FB_OP_ZTBMV',
    'FB_OP_STBSV','FB_OP_DTBSV','FB_OP_CTBSV','FB_OP_ZTBSV',
    'FB_OP_STPMV','FB_OP_DTPMV','FB_OP_CTPMV','FB_OP_ZTPMV',
    'FB_OP_STPSV','FB_OP_DTPSV','FB_OP_CTPSV','FB_OP_ZTPSV',
    'FB_OP_SSPR','FB_OP_DSPR','FB_OP_CHPR','FB_OP_ZHPR',
    'FB_OP_SSPR2','FB_OP_DSPR2','FB_OP_CHPR2','FB_OP_ZHPR2',
    # L3
    'FB_OP_SGEMM','FB_OP_DGEMM','FB_OP_CGEMM','FB_OP_ZGEMM',
    'FB_OP_SSYMM','FB_OP_DSYMM','FB_OP_CSYMM','FB_OP_ZSYMM',
    'FB_OP_CHEMM','FB_OP_ZHEMM',
    'FB_OP_SSYRK','FB_OP_DSYRK','FB_OP_CSYRK','FB_OP_ZSYRK',
    'FB_OP_CHERK','FB_OP_ZHERK',
    'FB_OP_SSYR2K','FB_OP_DSYR2K','FB_OP_CSYR2K','FB_OP_ZSYR2K',
    'FB_OP_CHER2K','FB_OP_ZHER2K',
    'FB_OP_STRMM','FB_OP_DTRMM','FB_OP_CTRMM','FB_OP_ZTRMM',
    'FB_OP_STRSM','FB_OP_DTRSM','FB_OP_CTRSM','FB_OP_ZTRSM',
]

# Verify all parsed entries are covered
missing = [s for s in entries if s not in order]
if missing:
    print(f"WARNING: these parsed entries are not in order list: {missing}")

# Build replacement table text
new_table_lines = []
section_comments = {
    'FB_OP_SAXPY':  '    /* BLAS Level 1 */',
    'FB_OP_SGEMV':  '    /* BLAS Level 2 */',
    'FB_OP_SGEMM':  '    /* BLAS Level 3 */',
}

for sym in order:
    if sym in section_comments:
        new_table_lines.append(section_comments[sym])
    runner = entries.get(sym, 'NULL')
    if runner == 'NULL':
        if sym in not_impl:
            new_table_lines.append(f'    [{sym}] = NULL,  /* not yet implemented */')
        # else: omit entirely — C zero-initializes to NULL
    else:
        new_table_lines.append(f'    [{sym}] = {runner},')

new_table_text = '\n'.join(new_table_lines) + '\n'

# Also update FB_DIRECT_DISPATCH_SIZE → use FB_JUDGE_MAX_OPERATIONS so the
# bounds check in the entry point works correctly after any renumbering.
# We'll update the #define and the array size declaration.

# Reconstruct the file
prefix = lines[:table_start]
suffix = lines[table_end:]

new_file = (
    ''.join(prefix)
    + new_table_text
    + ''.join(suffix)
)

# Also fix the #define FB_DIRECT_DISPATCH_SIZE and static array declaration
new_file = re.sub(
    r'#define FB_DIRECT_DISPATCH_SIZE\s+\d+',
    '/* Dispatch table spans all op IDs — size is FB_JUDGE_MAX_OPERATIONS */',
    new_file
)
new_file = re.sub(
    r'fb_direct_dispatch\[FB_DIRECT_DISPATCH_SIZE\]',
    'fb_direct_dispatch[FB_JUDGE_MAX_OPERATIONS]',
    new_file
)
# Also fix the bounds check in the entry point
new_file = re.sub(
    r'op_id >= FB_DIRECT_DISPATCH_SIZE',
    'op_id >= FB_JUDGE_MAX_OPERATIONS',
    new_file
)

with open(fname, 'w', encoding='utf-8', newline='\n') as f:
    f.write(new_file)

print(f"Done. Written {fname}")
print(f"Non-NULL entries emitted: {sum(1 for s in order if entries.get(s,'NULL') != 'NULL')}")
print(f"NULL/not-impl entries emitted: {len(not_impl)}")
print(f"Omitted (zero-init) NULL entries: {sum(1 for s in order if entries.get(s,'NULL') == 'NULL' and s not in not_impl)}")
