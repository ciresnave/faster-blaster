"""
apply_vtable_expansion.py
Applies generated expansions to:
  1. src/backends/backend_interface.h  — inserts 2029 fb_generic_fn fields
  2. src/core/op_vtable_map.c          — appends 2029 map entries
"""
import re, sys

# ── 1. Patch backend_interface.h ─────────────────────────────────────────────
VTABLE_HDR = "src/backends/backend_interface.h"
FIELDS_TXT = "build/vtable_new_fields.txt"

new_fields = open(FIELDS_TXT, encoding="utf-8").read().rstrip()

src = open(VTABLE_HDR, encoding="utf-8").read()

# Insert the new fields just before the ext_ops[] block comment
INSERT_BEFORE = "  /* =========================================================================\n   * Extended operation dispatch table (full 2266-op superset)"

if INSERT_BEFORE not in src:
    print("ERROR: anchor not found in backend_interface.h")
    sys.exit(1)

idx = src.index(INSERT_BEFORE)
src = src[:idx] + new_fields + "\n\n" + src[idx:]

open(VTABLE_HDR, "w", encoding="utf-8", newline="\n").write(src)
print(f"Patched {VTABLE_HDR}")

# ── 2. Patch op_vtable_map.c ─────────────────────────────────────────────────
MAP_SRC   = "src/core/op_vtable_map.c"
MAP_TXT   = "build/op_map_new_entries.txt"

new_entries = open(MAP_TXT, encoding="utf-8").read().rstrip()

src2 = open(MAP_SRC, encoding="utf-8").read()

# Insert new entries just before the array-closing  };\n\n#define K_OP_FIELD_MAP_COUNT
INSERT_BEFORE_MAP = "};\n\n#define K_OP_FIELD_MAP_COUNT"
if INSERT_BEFORE_MAP not in src2:
    print("ERROR: array-closing anchor not found in op_vtable_map.c")
    sys.exit(1)

idx2 = src2.index(INSERT_BEFORE_MAP)
src2 = src2[:idx2] + new_entries + "\n" + src2[idx2:]

open(MAP_SRC, "w", encoding="utf-8", newline="\n").write(src2)
print(f"Patched {MAP_SRC}")

print("Done.")
