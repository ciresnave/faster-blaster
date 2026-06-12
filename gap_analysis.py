"""
Produce the exact list of uncovered FB_OP_* operations (the 317 missing ones)
that faster-blaster-reference needs to implement.
"""
import re, subprocess, sys
from pathlib import Path

FB_ROOT   = Path(r"c:\Users\cires\OneDrive\Documents\projects\faster-blaster")
FBR_ROOT  = Path(r"c:\Users\cires\OneDrive\Documents\projects\faster-blaster-reference")
DUMPBIN   = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\arm64\dumpbin.exe"
DLL       = FBR_ROOT / "build-extended" / "libfaster_blaster_reference.dll"

# ---- Load stem table from op_vtable_map.c ----
text = (FB_ROOT / "src/core/op_vtable_map.c").read_text(encoding="utf-8", errors="replace")
pat  = re.compile(r'\{\s*"([^"]+)"\s*,\s*(FB_OP_\w+)\s*\}')
stem_table: dict[str,str] = {m.group(1): m.group(2) for m in pat.finditer(text)}

# ---- Load DLL exports ----
result = subprocess.run([DUMPBIN, "/EXPORTS", str(DLL)], capture_output=True, text=True)
exports: list[str] = []
for line in result.stdout.splitlines():
    m = re.match(r'^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)', line)
    if m: exports.append(m.group(1))

print(f"Stem table entries: {len(stem_table)}")
print(f"DLL exports: {len(exports)}")

# ---- Classify each export to a stem ----
def to_stem(sym: str) -> str:
    if sym.startswith("cblas_"):  return sym[6:]
    if sym.startswith("fb_"):     return sym
    if sym.endswith("_ref"):      return sym[:-4]
    if sym.endswith("_"):         return sym[:-1]
    return sym

covered: set[str] = set()
for sym in exports:
    # First: exact match (handles FB_OP_CBLAS_* entries like "cblas_sgemm_batch")
    if sym in stem_table:
        covered.add(stem_table[sym])
    # Then: stripped-stem match (handles cblas_saxpy -> "saxpy" -> FB_OP_SAXPY)
    s = to_stem(sym)
    if s in stem_table:
        covered.add(stem_table[s])

uncovered_ops = set(stem_table.values()) - covered

# Build reverse map: FB_OP_* -> [stems]
op_to_stems: dict[str, list[str]] = {}
for stem, op in stem_table.items():
    op_to_stems.setdefault(op, []).append(stem)

print(f"\nUncovered ops: {len(uncovered_ops)}")
print("=" * 70)

# Group by category
def categorize(op: str) -> str:
    core = op.replace("FB_OP_", "")
    if core.startswith("CBLAS_"):
        return "A_CBLAS_EXTENDED"
    # Use first 1-3 chars of the numeric prefix to identify BLAS vs LAPACK
    first = core[0] if core else "?"
    if first in "CDSZIL" and len(core) > 1:
        # Look at stems to identify
        stems = op_to_stems.get(op, [op])
        # If all stems are len ≤ 6, likely BLAS; longer = LAPACK
        avg_len = sum(len(s) for s in stems) / len(stems)
        if avg_len <= 8:
            if any("batch" in s or "strided" in s for s in stems):
                return "A_CBLAS_EXTENDED"
    # Check stems for LAPACK patterns
    stems = op_to_stems.get(op, [])
    for s in stems:
        if len(s) >= 5:
            suffix = s[1:] if s[0] in "sdzc" else s[2:]
            if any(suffix.startswith(p) for p in ["gsvx","esvxx","osvxx","esvx","psvx","tsvx","bsvx","bzsvx","bsvxx","ggsvd","gges","ggev","gglse","ggglm","eevr","evx","tbevd","tbevx","sbevd","sbevx","stebz","stegr","stein","laed","lasd","bdsdc","bdsqr","disna","larrc","larre","larrd"]):
                return "C_LAPACK_EXPERT"
            if any(suffix.startswith(p) for p in ["ggev","gges","ggevx","ggesx","bsvx","gelsd","gelsy","gtsvx","hpevd","hpevx","hpev","sbev","spev","stev","sygvd","sygvx","hegvd","hegvx","hpgvd","hpgvx","spgvd","spgvx"]):
                return "C_LAPACK_EXPERT"
    return "B_LAPACK_MISSING"

groups: dict[str, list[tuple[str, list[str]]]] = {}
for op in sorted(uncovered_ops):
    cat = categorize(op)
    stems = sorted(op_to_stems.get(op, []))
    groups.setdefault(cat, []).append((op, stems))

for cat, items in sorted(groups.items()):
    print(f"\n## {cat} ({len(items)} ops)")
    for op, stems in sorted(items):
        print(f"  {op:40} stems: {stems}")
