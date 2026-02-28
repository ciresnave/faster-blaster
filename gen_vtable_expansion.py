"""
gen_vtable_expansion.py — Generate vtable + map additions for all missing ops.
Generates:
  build/vtable_new_fields.txt   — fb_generic_fn lines to paste into vtable struct
  build/op_map_new_entries.txt  — k_op_field_map[] entries for op_vtable_map.c
"""
import re, os

OPS_HEADER = "src/judge/judge_op_ids.h"
VTABLE_HDR = "src/backends/backend_interface.h"

# Section ranges  (start_id, end_id_exclusive, label)
SECTIONS = [
    (  157, 1322, "LAPACK — standard s/d/c/z routines"),
    ( 1322, 1634, "ScaLAPACK — parallel p* distributed routines"),
    ( 1634, 1808, "Extended BLAS (cblas_batch / cblas_strided / axpby variants)"),
    ( 1808, 1851, "MKL extensions (mkl_jit, mkl_*omatcopy, mkl_sparse_*)"),
    ( 1851, 1904, "Deep Neural Network primitives (fb_dnn_*)"),
    ( 1904, 1943, "FFT (fb_fft_*)"),
    ( 1943, 2001, "Sparse linear algebra (fb_sparse_*)"),
    ( 2001, 2118, "Tensor operations + RNG + NCCL/RCCL collectives"),
    ( 2118, 2142, "Statistics + Machine Learning"),
    ( 2142, 2177, "Parallel Primitives (fb_prim_*)"),
    ( 2177, 2185, "Chemistry / Physics extensions"),
    ( 2185, 2218, "Vector math (fb_v*)"),
    ( 2218, 2234, "Spline interpolation (fb_spline_*)"),
    ( 2234, 2266, "GEMM / LPGEMM fused extensions"),
]

def section_for(op_id):
    for s, e, label in SECTIONS:
        if s <= op_id < e:
            return label
    return None

# ── Parse all FB_OP_* defs ────────────────────────────────────────────────────
ops = []
for l in open(OPS_HEADER, encoding="utf-8"):
    m = re.match(r"#define (FB_OP_([A-Z0-9_]+))\s+(\d+)", l.strip())
    if m and not m.group(2).startswith('_'):  # skip sentinel FB_OP__XXX_BEGIN/END
        ops.append((m.group(2).lower(), m.group(1), int(m.group(3))))

ops.sort(key=lambda x: x[2])

# ── Collect existing named vtable fields ──────────────────────────────────────
existing = set()
for l in open(VTABLE_HDR, encoding="utf-8"):
    m = re.match(r"\s+fb_(\w+)_fn\s+(\w+)\s*;", l)
    if m:
        existing.add(m.group(2))
    m2 = re.match(r"\s+fb_generic_fn\s+(\w+)\s*;", l)
    if m2:
        existing.add(m2.group(1))

missing = [(n, f, i) for n, f, i in ops if n not in existing]
print(f"Total ops: {len(ops)}, existing named: {len(existing)}, to add: {len(missing)}")

# ── Generate vtable field declarations ───────────────────────────────────────
os.makedirs("build", exist_ok=True)

out_fields = []
cur_section = None
for name, full_macro, op_id in missing:
    sec = section_for(op_id)
    if sec and sec != cur_section:
        cur_section = sec
        out_fields.append("")
        out_fields.append(f"  /* {'='*72}")
        out_fields.append(f"   * {sec}")
        out_fields.append(f"   * {'='*72}")
        out_fields.append( "   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;")
        out_fields.append( "   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */")
        out_fields.append("")
    out_fields.append(f"  fb_generic_fn {name};  /* {full_macro} = {op_id} */")

with open("build/vtable_new_fields.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_fields) + "\n")
print(f"  -> build/vtable_new_fields.txt  ({len(missing)} declarations)")

# ── Generate op_vtable_map entries ───────────────────────────────────────────
out_map = []
cur_section = None
for name, full_macro, op_id in missing:
    sec = section_for(op_id)
    if sec and sec != cur_section:
        cur_section = sec
        out_map.append(f"    /* {sec} */")
    out_map.append(f"    {{ {full_macro}, offsetof(fb_backend_vtable_t, {name}) }},")

with open("build/op_map_new_entries.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_map) + "\n")
print(f"  -> build/op_map_new_entries.txt  ({len(missing)} entries)")
