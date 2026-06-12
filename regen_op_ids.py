#!/usr/bin/env python3
"""
Regenerate judge_op_ids.h so IDs 314+ are family-first ordered:
  S/D/C/Z precision variants of the same base operation get consecutive IDs.
  Families are sorted alphabetically.
  IDs 0-313 are preserved exactly.
"""

import re
from collections import defaultdict

SRC = 'src/judge/judge_op_ids.h'
lines = open(SRC, encoding='utf-8').readlines()

# ----- 1. Parse all existing defines ----------------------------------------

all_defines = []   # (sym, id, line_index)
for i, l in enumerate(lines):
    m = re.match(r'#define (FB_OP_\w+)\s+(\d+)', l.strip())
    if m:
        all_defines.append((m.group(1), int(m.group(2)), i))

# Partition: keep 0-313 unchanged
keep_block = [(s, n, li) for s, n, li in all_defines if n <= 313]
regen_block = [(s, n, li) for s, n, li in all_defines if n >= 314]

print(f"Preserved (0-313): {len(keep_block)}")
print(f"To reorder (314+): {len(regen_block)}")

# Also find the FB_JUDGE_MAX_OPERATIONS define (not in all_defines because it
# doesn't start with FB_OP_)
max_ops_line = None
for i, l in enumerate(lines):
    if l.strip().startswith('#define FB_JUDGE_MAX_OPERATIONS'):
        max_ops_line = i
        break

# ----- 2. Extract block-level comments from the header ----------------------
# The file uses section headers like:
#   /* ===... LAPACK ... === */
# We need to find which comment blocks correspond to which ops so we can
# rebuild the section structure.  Strategy: keep the prefix unchanged (all
# lines up to and including the last "preserved" define), then regenerate
# only the "generated" portion.

# Find the last preserved define in the file and everything up to that line
last_keep_lineidx = max(li for _, _, li in keep_block)

# ----- 3. Group regen ops by family name ------------------------------------

PRECISION_ORDER = {'S': 0, 'D': 1, 'C': 2, 'Z': 3}

def precision_and_base(sym):
    """
    e.g.  FB_OP_SGETRF  ->  ('S', 'GETRF')
          FB_OP_CAXPY_BATCH  ->  ('C', 'AXPY_BATCH')
          FB_OP_FB_GEMM_UNIFIED  ->  (None, 'FB_GEMM_UNIFIED')   (no prefix)
          FB_OP_ILAENV  ->  (None, 'ILAENV')
    """
    name = sym[len('FB_OP_'):]  # strip FB_OP_
    if name and name[0] in 'SDCZ':
        prec = name[0]
        base = name[1:]
        # Guard: must have a non-trivial base (at least 2 chars) to count as
        # a precision-parameterised variant.  Prevents e.g. FB_OP_S -> ('S','')
        if len(base) >= 2:
            return prec, base
    return None, name

# Group symbols by base name
family_map = defaultdict(list)  # base -> [(prec_char_or_None, sym)]

for sym, _, _ in regen_block:
    prec, base = precision_and_base(sym)
    family_map[base].append((prec, sym))

# Sort within each family: (S,D,C,Z) first, then None (non-precision ops)
def prec_key(item):
    prec, sym = item
    if prec is None:
        return (4, sym)  # after S/D/C/Z
    return (PRECISION_ORDER[prec], sym)

for base in family_map:
    family_map[base].sort(key=prec_key)

# Sort families alphabetically by base name
sorted_families = sorted(family_map.items())

# ----- 4. Assign new IDs starting at 314 ------------------------------------

next_id = 314
regen_assigns = []  # (sym, new_id)

for base, members in sorted_families:
    for prec, sym in members:
        regen_assigns.append((sym, next_id))
        next_id += 1

print(f"New max ID: {next_id - 1}  (was {max(n for _,n,_ in regen_block)})")
assert next_id <= 3000, f"Overflow! next_id={next_id}"

# Check for any ops beyond the hand-written LAPACK block comment range that
# need explicit gap reservations in the header comment.  We won't add gaps
# now — the consecutive dense numbering is the whole point.

# ----- 5. Rebuild the file --------------------------------------------------

# Figure out which block-comment sections exist in the original regen portion
# We want to preserve the section structure.  Find all /*...=== */ or
# /* ==== ... */ comment lines between regen block start and end.

regen_start_line = min(li for _, _, li in regen_block)
regen_end_line   = max(li for _, _, li in regen_block)

# Build a mapping: define symbol -> (old comment line just before it, if any)
# We'll re-use section headers verbatim from the original file.

# Collect section header lines in the regen portion
section_headers = []  # (line_index, text)
for i in range(regen_start_line, regen_end_line + 1):
    l = lines[i].rstrip()
    if '=====' in l and '/*' in l:
        section_headers.append((i, l))

# Build a map: first_define_after_each_header -> header_text
# We'll emit the header just before the first symbol in each section.

# To do this properly: find which define each header precedes.
# Sort section headers by line index.
# For each define, if there's a header between it and the previous define, emit it.

define_to_header = {}
sorted_regen_by_line = sorted(regen_block, key=lambda x: x[2])  # by line index
header_iter = iter(section_headers)
cur_header = next(header_iter, None)
for idx, (sym, n, li) in enumerate(sorted_regen_by_line):
    if cur_header and cur_header[0] < li:
        # This header appears before this define
        define_to_header[sym] = cur_header[1]
        cur_header = next(header_iter, None)

# Build new regen text
regen_sym_to_new_id = dict(regen_assigns)
old_sym_order = [sym for sym, _, _ in sorted_regen_by_line]
# Now we need to emit in the NEW family-first order.
# For section headers, we reuse the original headers at their original positions
# relative to the new order.  Simple approach: emit headers at family-boundaries
# that correspond to the original sections.

# Actually the cleanest approach: just emit all new assigns with minimal section
# markers.  We'll keep the same major section structure by detecting when
# the base ID range crosses a section boundary in the original file.

# Let's just group by original section (using the headers we found):
# Match each symbol to its original section header.

def build_new_regen():
    sections = []  # [(header_or_None, [(sym, new_id)])]
    cur_section_header = None
    cur_section_syms = []

    # Walk sorted_regen_by_line and group by section header
    for sym, n, li in sorted_regen_by_line:
        h = define_to_header.get(sym)
        if h:
            if cur_section_syms:
                sections.append((cur_section_header, cur_section_syms))
            cur_section_header = h
            cur_section_syms = []
        cur_section_syms.append(sym)
    if cur_section_syms:
        sections.append((cur_section_header, cur_section_syms))

    # Build set of syms per section (original order)
    sym_to_section = {}
    for hdr, syms in sections:
        for s in syms:
            sym_to_section[s] = hdr

    # Now emit family-first order, but inject section headers when we first
    # encounter a symbol from a new section.
    out_lines = []
    current_section_emitted = None
    for sym, new_id in regen_assigns:
        sec = sym_to_section.get(sym)
        if sec and sec != current_section_emitted:
            out_lines.append('')
            out_lines.append(sec.rstrip())
            out_lines.append('' )
            current_section_emitted = sec
        out_lines.append(f'#define {sym:<44} {new_id}')
    return out_lines

new_regen_lines = build_new_regen()

# ----- 6. Assemble the full file --------------------------------------------

# Everything up to (and including) the last preserved define line
prefix = lines[:last_keep_lineidx + 1]

# But: there may be blank lines after the last preserved define and before
# the next section header.  Include them.
end_prefix = last_keep_lineidx + 1
while end_prefix < len(lines) and lines[end_prefix].strip() == '':
    end_prefix += 1

# The suffix: everything after the last regen define (the #endif etc.)
suffix_start = regen_end_line + 1
while suffix_start < len(lines) and lines[suffix_start].strip() == '':
    suffix_start += 1
suffix = lines[suffix_start:]

new_file = (
    ''.join(prefix)
    + '\n'
    + '\n'.join(new_regen_lines)
    + '\n'
    + '\n'
    + ''.join(suffix)
)

# Final sanity check: count defines
defines_in_output = re.findall(r'^#define FB_OP_\w+\s+\d+', new_file, re.MULTILINE)
print(f"Defines in output: {len(defines_in_output)}")
duplicates = [x for x in defines_in_output if defines_in_output.count(x) > 1]
if duplicates:
    print(f"WARNING: duplicates found: {set(duplicates)[:5]}")

with open(SRC, 'w', encoding='utf-8', newline='\n') as f:
    f.write(new_file)

print(f"Written {SRC}")
print(f"New ID range: 314 – {next_id - 1}")
