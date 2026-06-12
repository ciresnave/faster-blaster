#!/usr/bin/env python3
"""
vtable_audit.py — Simulates fb_classify_symbol() + fb_install_conv_thunks()
against the faster-blaster-reference DLL exports to report vtable coverage.

Usage:
    python vtable_audit.py

Outputs:
    - # ops covered by CBLAS slot (from DLL cblas_* / fb_* / LAPACKE_*)
    - # ops covered by Fortran slot (from DLL trailing-underscore exports)
    - # ops with both slots (CBLAS directly) — thunks make full coverage trivially
    - # ops with only CBLAS (thunks provide Fortran if in thunk list)
    - # ops with only Fortran (thunks provide CBLAS if in thunk list)
    - # ops with neither slot covered (genuine gaps)
"""

import re
import subprocess
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
FB_ROOT = Path(__file__).parent
REF_ROOT = FB_ROOT.parent / "faster-blaster-reference"

OP_IDS_HEADER    = FB_ROOT / "src" / "judge" / "judge_op_ids.h"
VTABLE_MAP_C     = FB_ROOT / "src" / "core" / "op_vtable_map.c"
CONV_THUNKS_C    = FB_ROOT / "src" / "core" / "conv_thunks.c"
REF_DLL          = REF_ROOT / "build-extended" / "libfaster_blaster_reference.dll"

# ─── 1. Parse judge_op_ids.h → {name: id} and {id: name} ─────────────────────
def parse_op_ids(path: Path):
    id_by_name = {}   # "FB_OP_SAXPY" → 42
    name_by_id = {}   # 42 → "FB_OP_SAXPY"
    # Format: #define FB_OP_SAXPY   42
    # Skip double-underscore sentinels like FB_OP__BLAS_L1_BEGIN
    pattern = re.compile(r'^\s*#define\s+(FB_OP_\w+)\s+(\d+)', re.MULTILINE)
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in pattern.finditer(text):
        name = m.group(1)
        if name.startswith("FB_OP__"):   # skip section sentinels
            continue
        val  = int(m.group(2))
        id_by_name[name] = val
        name_by_id[val]  = name
    return id_by_name, name_by_id

# ─── 2. Parse k_op_stem_map[] in op_vtable_map.c → {stem: op_id_int} ─────────
def parse_stem_map(path: Path, id_by_name: dict):
    stem_map = {}  # "saxpy" → 42
    text = path.read_text(encoding="utf-8", errors="replace")

    # Find the k_op_stem_map block
    start = text.find("k_op_stem_map[]")
    if start == -1:
        sys.exit("ERROR: k_op_stem_map[] not found in op_vtable_map.c")
    block_start = text.find("{", start)

    # Scan entries until the closing "};" of the array
    entry_pattern = re.compile(
        r'\{\s*"([^"]+)"\s*,\s*(FB_OP_\w+)\s*\}'
    )
    # Walk from block_start, track brace depth to find array end
    depth   = 0
    in_array = False
    end_pos  = len(text)
    for i in range(block_start, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
            in_array = True
        elif c == '}':
            depth -= 1
            if in_array and depth == 0:
                end_pos = i + 1
                break

    block = text[block_start:end_pos]
    for m in entry_pattern.finditer(block):
        stem   = m.group(1)
        op_sym = m.group(2)
        op_id  = id_by_name.get(op_sym)
        if op_id is not None:
            stem_map[stem] = op_id
    return stem_map

# ─── 3. Get DLL exports via dumpbin or llvm-objdump ──────────────────────────
def get_dll_exports(dll_path: Path):
    """Return list of exported symbol names from the DLL."""
    # Try dumpbin first (MSVC/VS), then llvm-objdump, then nm-style
    exports = []

    # Method A: dumpbin /EXPORTS
    try:
        result = subprocess.run(
            ["dumpbin", "/EXPORTS", str(dll_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            # Lines like: "   1    0 00001234 cblas_saxpy"
            pat = re.compile(r'^\s+\d+\s+\w+\s+\w+\s+(\S+)', re.MULTILINE)
            for m in pat.finditer(result.stdout):
                sym = m.group(1)
                if sym not in ("name", "RVA", "ordinal"):
                    exports.append(sym)
            if exports:
                print(f"[dumpbin] Got {len(exports)} exports")
                return exports
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method B: llvm-objdump -p
    for objdump_cmd in ("llvm-objdump", "llvm-objdump-18", "llvm-objdump-17"):
        try:
            result = subprocess.run(
                [objdump_cmd, "-p", str(dll_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                # In PE export table output: "    name ....   saxpy_"
                # Look for lines after "Export Table:"
                in_exports = False
                for line in result.stdout.splitlines():
                    if "Export Table" in line:
                        in_exports = True
                        continue
                    if in_exports:
                        m = re.search(r'\]\s+(\S+)$', line)
                        if m:
                            exports.append(m.group(1))
                        elif re.match(r'^\s{0,2}\S', line) and in_exports and not line.strip():
                            break
                if exports:
                    print(f"[{objdump_cmd}] Got {len(exports)} exports")
                    return exports
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Method C: Python ctypes — walk the PE export directory
    try:
        exports = _get_exports_via_ctypes(dll_path)
        if exports:
            print(f"[ctypes/PE] Got {len(exports)} exports")
            return exports
    except Exception as e:
        print(f"[ctypes] Failed: {e}")

    sys.exit(f"ERROR: Could not get exports from {dll_path}. Ensure dumpbin or llvm-objdump is on PATH.")

def _get_exports_via_ctypes(dll_path: Path):
    """Walk the PE export directory directly using ctypes."""
    import ctypes, ctypes.wintypes as wt
    import struct

    with open(dll_path, "rb") as f:
        data = f.read()

    # DOS header → PE offset
    if data[:2] != b'MZ':
        raise ValueError("Not a PE file")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    sig = data[pe_off:pe_off+4]
    if sig != b'PE\0\0':
        raise ValueError("PE signature not found")

    # Optional header
    coff_off = pe_off + 4
    machine  = struct.unpack_from("<H", data, coff_off)[0]
    opt_off  = coff_off + 20
    magic    = struct.unpack_from("<H", data, opt_off)[0]

    if magic == 0x20b:  # PE32+
        export_dd_off = opt_off + 112
    elif magic == 0x10b:  # PE32
        export_dd_off = opt_off + 96
    else:
        raise ValueError(f"Unknown PE magic 0x{magic:x}")

    export_rva, export_size = struct.unpack_from("<II", data, export_dd_off)
    if export_rva == 0 or export_size == 0:
        return []

    # Resolve RVA to file offset via section table
    num_sections = struct.unpack_from("<H", data, coff_off + 2)[0]
    size_of_optional = struct.unpack_from("<H", data, coff_off + 16)[0]
    sections_off = opt_off + size_of_optional

    def rva_to_offset(rva):
        for i in range(num_sections):
            s = sections_off + i * 40
            virt_addr  = struct.unpack_from("<I", data, s + 12)[0]
            virt_size  = struct.unpack_from("<I", data, s + 16)[0]
            raw_off    = struct.unpack_from("<I", data, s + 20)[0]
            if virt_addr <= rva < virt_addr + virt_size:
                return raw_off + (rva - virt_addr)
        raise ValueError(f"RVA 0x{rva:x} not found in any section")

    exp_off = rva_to_offset(export_rva)

    # IMAGE_EXPORT_DIRECTORY (40 bytes)
    num_names     = struct.unpack_from("<I", data, exp_off + 24)[0]
    names_rva     = struct.unpack_from("<I", data, exp_off + 32)[0]
    names_off     = rva_to_offset(names_rva)

    symbols = []
    for i in range(num_names):
        name_rva = struct.unpack_from("<I", data, names_off + i * 4)[0]
        name_off = rva_to_offset(name_rva)
        end = data.index(b'\x00', name_off)
        symbols.append(data[name_off:end].decode("ascii", errors="replace"))

    return symbols

# ─── 4. Parse conv_thunks.c: find which op stems have CBLAS↔Fortran thunks ───
def parse_thunk_ops(path: Path, stem_map: dict):
    """Return set of FB_OP_* name strings that have thunks."""
    # The arrays k_cblas_to_fortran_thunks and k_fortran_to_cblas_thunks
    # have designated-init entries: [FB_OP_SAXPY] = C2F(...)
    text = path.read_text(encoding="utf-8", errors="replace")
    pat  = re.compile(r'\[(FB_OP_[A-Z0-9_]+)\]\s*=')
    thunked_names = set()
    for m in pat.finditer(text):
        thunked_names.add(m.group(1))
    return thunked_names  # set of "FB_OP_*" string names

# ─── 5. Simulate fb_classify_symbol() ────────────────────────────────────────
FB_CONV_CBLAS   = 0
FB_CONV_FORTRAN = 1
FB_CONV_COUNT   = 2  # unmatched

def classify_symbol(name: str, stem_map: dict):
    """
    Mirrors src/core/backend_auto_detect.c::fb_classify_symbol()
    INCLUDING the two fixes applied to backend_auto_detect.c:
      1. cblas_* branch also tries full name (fixes 94 FB_OP_CBLAS_* ops)
      2. Plain-stem fallback before the final return (fixes sgemm_ptr etc.)
    Returns (op_id: int, conv: int) or (None, FB_CONV_COUNT).
    """
    n = len(name)

    # cblas_<stem> — try stripped, then full name (fixed)
    if n > 6 and name.startswith("cblas_"):
        stem = name[6:]
        op_id = stem_map.get(stem)
        if op_id is not None:
            return op_id, FB_CONV_CBLAS
        # Fallback: try full name (for cblas_cgemm_batch_strided etc.)
        op_id = stem_map.get(name)
        if op_id is not None:
            return op_id, FB_CONV_CBLAS
        return None, FB_CONV_COUNT

    # fb_<stem> — try stripped, then full
    if n > 3 and name.startswith("fb_"):
        stem = name[3:]
        op_id = stem_map.get(stem)
        if op_id is not None:
            return op_id, FB_CONV_CBLAS
        op_id = stem_map.get(name)  # full "fb_..." key
        if op_id is not None:
            return op_id, FB_CONV_CBLAS
        return None, FB_CONV_COUNT

    # LAPACKE_<stem>
    if n > 8 and name.startswith("LAPACKE_"):
        stem = name[8:]
        op_id = stem_map.get(stem)
        if op_id is not None:
            return op_id, FB_CONV_CBLAS
        return None, FB_CONV_COUNT

    # <stem>_ — Fortran trailing underscore
    if n > 1 and name.endswith("_") and n < 70:
        stem = name[:-1]
        op_id = stem_map.get(stem)
        if op_id is not None:
            return op_id, FB_CONV_FORTRAN
        return None, FB_CONV_COUNT

    # Plain stem fallback — C-ABI exports with no prefix (sgemm_ptr, etc.)
    op_id = stem_map.get(name)
    if op_id is not None:
        return op_id, FB_CONV_CBLAS

    return None, FB_CONV_COUNT

# ─── 6. Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("faster-blaster vtable coverage audit")
    print("=" * 70)

    # Load data
    print(f"\n[1/4] Parsing {OP_IDS_HEADER.name} ...")
    id_by_name, name_by_id = parse_op_ids(OP_IDS_HEADER)
    total_ops = len(id_by_name)
    max_id    = max(id_by_name.values()) if id_by_name else 0
    print(f"      {total_ops} FB_OP_* constants, max id = {max_id}")

    print(f"\n[2/4] Parsing {VTABLE_MAP_C.name} stem map ...")
    stem_map = parse_stem_map(VTABLE_MAP_C, id_by_name)
    print(f"      {len(stem_map)} stem→op_id entries")

    print(f"\n[3/4] Reading DLL exports from {REF_DLL.name} ...")
    exports = get_dll_exports(REF_DLL)
    print(f"      {len(exports)} exported symbols")

    print(f"\n[4/4] Parsing thunk coverage in {CONV_THUNKS_C.name} ...")
    thunk_op_names = parse_thunk_ops(CONV_THUNKS_C, stem_map)
    thunk_op_ids   = {id_by_name[n] for n in thunk_op_names if n in id_by_name}
    print(f"      {len(thunk_op_ids)} ops have CBLAS↔Fortran thunks installed")

    # ── Classify every export ───────────────────────────────────────────────
    print("\n─── Classifying exports ───────────────────────────────────────────")

    cblas_covered   = {}  # op_id → symbol name
    fortran_covered = {}  # op_id → symbol name
    unmatched       = []  # symbols that don't map to any op

    by_prefix = {"cblas_": 0, "fb_": 0, "LAPACKE_": 0, "fortran_": 0,
                 "other": 0}

    for sym in exports:
        op_id, conv = classify_symbol(sym, stem_map)
        if conv == FB_CONV_CBLAS:
            if op_id not in cblas_covered:
                cblas_covered[op_id] = sym
            if sym.startswith("cblas_"):    by_prefix["cblas_"] += 1
            elif sym.startswith("fb_"):     by_prefix["fb_"] += 1
            elif sym.startswith("LAPACKE_"):by_prefix["LAPACKE_"] += 1
            else:                           by_prefix["other"] += 1
        elif conv == FB_CONV_FORTRAN:
            if op_id not in fortran_covered:
                fortran_covered[op_id] = sym
            by_prefix["fortran_"] += 1
        else:
            unmatched.append(sym)

        # Simulate fill_slot dual-fill: for cblas_* that also have a full-name key
        if sym.startswith("cblas_"):
            alt_op_id = stem_map.get(sym)  # full name as key
            if alt_op_id is not None and alt_op_id != op_id:
                if alt_op_id not in cblas_covered:
                    cblas_covered[alt_op_id] = sym

    print(f"  Recognised as CBLAS conv: {sum(1 for _ in cblas_covered)} unique ops")
    print(f"    • cblas_* exports → CBLAS: {by_prefix['cblas_']}")
    print(f"    • fb_* exports → CBLAS:    {by_prefix['fb_']}")
    print(f"    • LAPACKE_* → CBLAS:       {by_prefix['LAPACKE_']}")
    print(f"  Recognised as Fortran conv: {sum(1 for _ in fortran_covered)} unique ops")
    print(f"    • trailing-_ exports:      {by_prefix['fortran_']}")
    print(f"  Unmatched exports:           {len(unmatched)}")

    # ── Per-op coverage ──────────────────────────────────────────────────────
    all_op_ids = set(name_by_id.keys())

    both_direct    = all_op_ids & set(cblas_covered) & set(fortran_covered)
    cblas_only     = (all_op_ids & set(cblas_covered)) - set(fortran_covered)
    fortran_only   = (all_op_ids & set(fortran_covered)) - set(cblas_covered)
    neither        = all_op_ids - set(cblas_covered) - set(fortran_covered)

    # After thunks: cblas_only ops that are in thunk_op_ids get fortran thunk
    #               fortran_only ops that are in thunk_op_ids get cblas thunk
    cblas_gains_fortran_thunk = cblas_only & thunk_op_ids
    fortran_gains_cblas_thunk = fortran_only & thunk_op_ids

    fully_covered_after_thunks = (
        both_direct |
        (cblas_only - cblas_gains_fortran_thunk) |   # cblas_only without thunk → still cblas only
        cblas_gains_fortran_thunk |                   # now both
        fortran_gains_cblas_thunk |                   # now both
        (fortran_only - fortran_gains_cblas_thunk)    # fortran_only without thunk → still fortran only
    )
    # "fully covered" = has at least one conv filled
    has_any_direct   = set(cblas_covered) | set(fortran_covered)
    has_cblas_after  = set(cblas_covered) | fortran_gains_cblas_thunk
    has_fortran_after= set(fortran_covered) | cblas_gains_fortran_thunk
    has_both_after   = has_cblas_after & has_fortran_after

    print("\n─── Vtable coverage summary ───────────────────────────────────────")
    print(f"  Total FB_OP_* operations:          {total_ops}")
    print()
    print("  BEFORE thunks (direct DLL exports only):")
    print(f"    Both CBLAS+Fortran slots filled:   {len(both_direct)}")
    print(f"    CBLAS slot only:                   {len(cblas_only)}")
    print(f"    Fortran slot only:                 {len(fortran_only)}")
    print(f"    Neither slot filled:               {len(neither)}")
    print(f"    At least one slot filled:          {len(has_any_direct)}")
    pct_direct = 100 * len(has_any_direct) / total_ops if total_ops else 0
    print(f"    Coverage (≥1 slot):               {pct_direct:.1f}%")
    print()
    print("  AFTER fb_install_conv_thunks():")
    print(f"    Ops gaining Fortran thunk:         {len(cblas_gains_fortran_thunk)}")
    print(f"    Ops gaining CBLAS thunk:           {len(fortran_gains_cblas_thunk)}")
    remaining_gaps = neither
    pct_after = 100 * (total_ops - len(remaining_gaps)) / total_ops if total_ops else 0
    print(f"    Remaining gaps (no slot at all):   {len(remaining_gaps)}")
    print(f"    Coverage after thunks:            {pct_after:.1f}%")
    print(f"    Both slots after thunks:           {len(has_both_after)}")

    # ── Sample unmatched exports ─────────────────────────────────────────────
    print("\n─── Sample unmatched exports (first 40) ───────────────────────────")
    # Categorise them
    unmatched_cblas_miss = [s for s in unmatched if s.startswith("cblas_")]
    unmatched_fb_miss    = [s for s in unmatched if s.startswith("fb_")]
    unmatched_fortran_miss = [s for s in unmatched if s.endswith("_") and not s.startswith(("cblas_","fb_","LAPACKE_"))]
    unmatched_other      = [s for s in unmatched if s not in unmatched_cblas_miss + unmatched_fb_miss + unmatched_fortran_miss]

    print(f"  cblas_* with no stem-map entry:      {len(unmatched_cblas_miss)}")
    for s in sorted(unmatched_cblas_miss)[:10]:
        print(f"    {s}")
    print(f"  fb_* with no stem-map entry:         {len(unmatched_fb_miss)}")
    for s in sorted(unmatched_fb_miss)[:10]:
        print(f"    {s}")
    print(f"  Fortran trailing-_ no stem entry:    {len(unmatched_fortran_miss)}")
    for s in sorted(unmatched_fortran_miss)[:10]:
        print(f"    {s}")
    print(f"  Other unmatched:                     {len(unmatched_other)}")
    for s in sorted(unmatched_other)[:10]:
        print(f"    {s}")

    # ── Gap report ───────────────────────────────────────────────────────────
    print("\n─── Operations with NO coverage (neither slot) ────────────────────")
    if not neither:
        print("  ✅ ALL operations have at least one vtable slot filled!")
    else:
        print(f"  {len(neither)} ops have no coverage:")
        # Group by category from name
        by_cat = {}
        for oid in sorted(neither):
            n = name_by_id.get(oid, f"<id={oid}>")
            # Use FB_OP_ prefix style as crude category
            parts = n.replace("FB_OP_", "").split("_")
            cat = parts[0] if parts else "UNKNOWN"
            by_cat.setdefault(cat, []).append((oid, n))
        for cat in sorted(by_cat):
            entries = by_cat[cat]
            print(f"\n  [{cat}] ({len(entries)} ops)")
            for oid, name in entries[:20]:
                print(f"    {oid:4d}  {name}")
            if len(entries) > 20:
                print(f"    ... and {len(entries)-20} more")

    # ── Fortran-only ops without thunks ──────────────────────────────────────
    fortran_only_no_thunk = fortran_only - fortran_gains_cblas_thunk
    if fortran_only_no_thunk:
        print(f"\n─── Fortran-only ops without CBLAS thunk ({len(fortran_only_no_thunk)}) ─────────")
        print("  (accessible only via Fortran ABI after DLL load)")
        for oid in sorted(fortran_only_no_thunk)[:20]:
            print(f"    {oid:4d}  {name_by_id.get(oid,'?'):<50}  ← {fortran_covered[oid]}")
        if len(fortran_only_no_thunk) > 20:
            print(f"  ... and {len(fortran_only_no_thunk)-20} more")

    # ── CBLAS-only ops without thunks ────────────────────────────────────────
    cblas_only_no_thunk = cblas_only - cblas_gains_fortran_thunk
    if cblas_only_no_thunk:
        print(f"\n─── CBLAS-only ops without Fortran thunk ({len(cblas_only_no_thunk)}) ─────────")
        print("  (accessible only via CBLAS/C ABI after DLL load)")

    print("\n" + "=" * 70)
    print("Audit complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
