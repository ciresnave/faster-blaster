#!/usr/bin/env python3
"""
gen_stem_map.py
---------------
Reads judge_op_ids.h, extracts every FB_OP_<NAME> constant, derives the
lowercase stem, and emits a complete replacement for the k_op_stem_map[]
block in op_vtable_map.c.

Stems are sorted lexicographically (required for bsearch in fb_stem_to_op_id).
"""

import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
IDS_H      = ROOT / "src" / "judge" / "judge_op_ids.h"
VTABLE_C   = ROOT / "src" / "core" / "op_vtable_map.c"

# ── Parse judge_op_ids.h ───────────────────────────────────────────────────────
# Match:  #define FB_OP_SAXPY   4
# Skip:   #define FB_OP__BLAS_L1_BEGIN   0  (double-underscore range markers)
OP_PAT = re.compile(r'^#define\s+FB_OP_([A-Z][A-Z0-9_]+)\s+(\d+)')

ops = {}   # stem → (macro_name, id_value)

with open(IDS_H) as fh:
    for line in fh:
        m = OP_PAT.match(line.strip())
        if not m:
            continue
        raw_name = m.group(1)
        id_val   = int(m.group(2))
        # Skip range sentinels that start with an underscore (unlikely here
        # since we matched [A-Z] first, but also skip FB_OP__* entirely —
        # those are range markers, not real ops)
        if raw_name.startswith('_'):
            continue
        stem = raw_name.lower()
        macro = f"FB_OP_{raw_name}"
        ops[stem] = (macro, id_val)

print(f"[gen_stem_map] Parsed {len(ops)} op stems from {IDS_H.name}", file=sys.stderr)

# ── Build sorted entries ───────────────────────────────────────────────────────
sorted_items = sorted(ops.items())   # sorted by stem string

# Maximum stem length for alignment
max_stem = max(len(s) for s in ops)
max_macro = max(len(m) for m, _ in ops.values())

# ── Generate C fragment ────────────────────────────────────────────────────────
lines = []
# Group by LAPACK subsection, BLAS level, etc. — simpler: just emit with comments
# at section boundaries (detect by ID ranges from the header)

# ID ranges from judge_op_ids.h
RANGES = [
    (  0,   49, "BLAS Level 1"),
    ( 49,  119, "BLAS Level 2"),
    (119,  157, "BLAS Level 3"),
    (157, 1322, "LAPACK"),
    (1322,1634, "ScaLAPACK"),
    (1634,2400, "Extended / fb_*"),
]

def range_for(id_val):
    for (lo, hi, name) in RANGES:
        if lo <= id_val < hi:
            return name
    return "Unknown"

current_section = None
for stem, (macro, id_val) in sorted_items:
    sec = range_for(id_val)
    if sec != current_section:
        current_section = sec
        lines.append(f"    /* ---- {sec} {'-' * (55 - len(sec))} */")
    # Align columns: padding goes OUTSIDE the quoted string
    stem_col  = f'"{stem}"'           # exact string — no spaces inside
    macro_col = macro
    pad = max_stem - len(stem) + 4    # blanks between stem and macro
    lines.append(f'    {{ {stem_col},{" " * pad}{macro_col} }},')

fragment = "\n".join(lines)

# ── Patch op_vtable_map.c ─────────────────────────────────────────────────────
text = VTABLE_C.read_text(encoding='utf-8')

# Locate the array start and end
start_marker = "static const fb_stem_entry_t k_op_stem_map[] = {"
end_marker   = "};"

start_pos = text.find(start_marker)
if start_pos == -1:
    print("ERROR: could not find k_op_stem_map[] array start", file=sys.stderr)
    sys.exit(1)

# Find the matching closing brace+semicolon AFTER the array start
# Walk forward: count braces
depth = 0
i = start_pos
while i < len(text):
    if text[i] == '{':
        depth += 1
    elif text[i] == '}':
        depth -= 1
        if depth == 0:
            break
    i += 1
# i now points at the closing '}'
end_pos = i + 2  # include "};"

old_block = text[start_pos:end_pos]
new_block = f"{start_marker}\n{fragment}\n}};"

new_text = text[:start_pos] + new_block + text[end_pos:]

VTABLE_C.write_text(new_text, encoding='utf-8')
print(f"[gen_stem_map] Wrote {len(sorted_items)} stem entries to {VTABLE_C}", file=sys.stderr)
print(f"[gen_stem_map] Old block length: {len(old_block)} chars  New: {len(new_block)} chars", file=sys.stderr)
print("OK", flush=True)
