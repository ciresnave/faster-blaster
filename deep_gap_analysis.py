"""
Deep gap analysis: compare DLL exports vs vtable, search source for alternate impls.
"""
import re, os, collections

REF_ROOT = r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster-reference'
FB_ROOT  = r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster'

# ---------- 1. Load vtable stems ----------
with open(os.path.join(FB_ROOT, r'src\core\op_vtable_map.c'), encoding='utf-8', errors='replace') as f:
    content = f.read()
stem_start = content.find('static const fb_stem_entry_t k_op_stem_map')
stem_end   = content.find('};', stem_start) + 2
all_stems  = re.findall(r'"([^"]+)"', content[stem_start:stem_end])

SKIP_RE = re.compile(
    r'^p[a-z]|^mkl_|^dnn_|^conv|^pool|^batch_norm|^layer_norm|^relu_'
    r'|^sparse_|^spmv|^spmm|^fft|^ifft|^vd_|^vs_|^vz_|^vc_'
    r'|^tensor|^nccl|^reduce_|^allreduce|^fb_tensor|^gemm_unified'
)
blas_stems = [s for s in all_stems if not SKIP_RE.match(s)]

# ---------- 2. Load DLL exports ----------
exp_path = os.path.join(os.environ['TEMP'], 'exports2.txt')
with open(exp_path, encoding='utf-8', errors='replace') as f:
    exp_lines = f.readlines()

dll_cblas   = set()
dll_fortran = set()
dll_ref     = set()
for line in exp_lines:
    m = re.match(r'^\s+\d+\s+0x[0-9a-fA-F]+\s+(\S+)', line)
    if not m:
        continue
    sym = m.group(1)
    if sym.startswith('cblas_'):
        dll_cblas.add(sym[6:])
    elif sym.endswith('_ref') and not sym.startswith('cblas_'):
        dll_ref.add(sym[:-4])
    elif sym.endswith('_') and not sym.endswith('__'):
        dll_fortran.add(sym[:-1])

dll_stems = dll_cblas | dll_fortran
missing = [s for s in blas_stems if s not in dll_stems]

print(f'BLAS/LAPACK stems in vtable   : {len(blas_stems)}')
print(f'Matched by DLL (cblas/fortran): {len(blas_stems) - len(missing)}')
print(f'MISSING from DLL              : {len(missing)}')

has_ref_in_dll = [s for s in missing if s in dll_ref]
no_ref_at_all  = [s for s in missing if s not in dll_ref]
print(f'  Has *_ref in DLL already    : {len(has_ref_in_dll)}')
print(f'  Nothing in DLL at all       : {len(no_ref_at_all)}')
print()

# ---------- 3. Build source definition index (targeted) ----------
print('Scanning source files for function definitions...')
src_dir = os.path.join(REF_ROOT, 'src')

# We build stem -> set of (relpath, sym_name)
stem_index = collections.defaultdict(set)

def_re = re.compile(
    r'^[ \t]*(?:(?:static|inline|FBR_EXPORT)\s+)*'
    r'(?:void|int|float|double|long double|BLAS_L\w*|LAPACK_\w*)\s+'
    r'(?:FBR_EXPORT\s+)?(\w+)\s*\(',
    re.MULTILINE
)

src_files = []
for dirpath, dirnames, filenames in os.walk(src_dir):
    dirnames[:] = [d for d in dirnames if not d.startswith('build')]
    for fn in filenames:
        if fn.endswith('.c'):
            src_files.append(os.path.join(dirpath, fn))

print(f'  {len(src_files)} .c files to scan')

for fpath in src_files:
    relpath = fpath[len(REF_ROOT)+1:].replace('\\', '/')
    try:
        with open(fpath, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception:
        continue
    for m in def_re.finditer(text):
        sym = m.group(1)
        if sym.startswith('cblas_'):
            stem = sym[6:]
        elif sym.endswith('_ref'):
            stem = sym[:-4]
        elif sym.startswith('fb_'):
            stem = sym[3:]
        elif sym.endswith('_') and len(sym) > 3:
            stem = sym[:-1]
        else:
            stem = sym
        if stem and len(stem) >= 4:
            stem_index[stem].add((relpath, sym))

print(f'  Index contains {len(stem_index)} unique stems')
print()

# ---------- 4. Classify the "no_ref_at_all" group ----------
cat2_cblas_unexported = []
cat3_ref_in_source    = []
cat4_plain_in_source  = []
cat5_absent           = []

for stem in sorted(no_ref_at_all):
    hits = stem_index.get(stem, set())
    ref_hits   = [(f, s) for f, s in hits if s == stem + '_ref']
    cblas_hits = [(f, s) for f, s in hits if s == 'cblas_' + stem]
    plain_hits = [(f, s) for f, s in hits if s == stem]
    fort_hits  = [(f, s) for f, s in hits if s == stem + '_']

    if cblas_hits:
        cat2_cblas_unexported.append((stem, sorted(cblas_hits)[0]))
    elif ref_hits:
        cat3_ref_in_source.append((stem, sorted(ref_hits)[0]))
    elif plain_hits or fort_hits:
        best = sorted(plain_hits or fort_hits)[0]
        cat4_plain_in_source.append((stem, best))
    else:
        cat5_absent.append(stem)

# ---------- 5. Print report ----------
SEP = '=' * 70

print(SEP)
print(f'CAT 1 [{len(has_ref_in_dll):4d}]: *_ref already in DLL — just need cblas_ wrapper')
print(SEP)
for s in sorted(has_ref_in_dll):
    print(f'  {s}')

print()
print(SEP)
print(f'CAT 2 [{len(cat2_cblas_unexported):4d}]: cblas_ defined in source but FBR_EXPORT missing')
print(SEP)
for stem, (f, s) in cat2_cblas_unexported:
    print(f'  {stem:38s} {s}  {f}')

print()
print(SEP)
print(f'CAT 3 [{len(cat3_ref_in_source):4d}]: *_ref in source, need cblas_ wrapper + FBR_EXPORT')
print(SEP)
for stem, (f, s) in cat3_ref_in_source:
    print(f'  {stem:38s} {s}  {f}')

print()
print(SEP)
print(f'CAT 4 [{len(cat4_plain_in_source):4d}]: plain/Fortran impl — need cblas_ wrapper')
print(SEP)
for stem, (f, s) in cat4_plain_in_source:
    print(f'  {stem:38s} {s}  {f}')

print()
print(SEP)
print(f'CAT 5 [{len(cat5_absent):4d}]: Truly absent — no source implementation found')
print(SEP)
for s in cat5_absent:
    print(f'  {s}')

# ---------- 6. Save TSV ----------
out = os.path.join(FB_ROOT, 'gap_detail.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('CAT\tSTEM\tSYMBOL\tFILE\n')
    for s in sorted(has_ref_in_dll):
        f.write(f'1_REF_IN_DLL\t{s}\t{s}_ref\t(DLL)\n')
    for stem, (fp, s) in cat2_cblas_unexported:
        f.write(f'2_CBLAS_UNEXPORTED\t{stem}\t{s}\t{fp}\n')
    for stem, (fp, s) in cat3_ref_in_source:
        f.write(f'3_REF_IN_SOURCE\t{stem}\t{s}\t{fp}\n')
    for stem, (fp, s) in cat4_plain_in_source:
        f.write(f'4_PLAIN_IN_SOURCE\t{stem}\t{s}\t{fp}\n')
    for s in cat5_absent:
        f.write(f'5_ABSENT\t{s}\t\t\n')

total_w = len(has_ref_in_dll) + len(cat2_cblas_unexported) + len(cat3_ref_in_source) + len(cat4_plain_in_source)
print()
print(SEP)
print('FINAL SUMMARY')
print(SEP)
print(f'Cat 1 (*_ref in DLL, need cblas wrap)   : {len(has_ref_in_dll):4d}')
print(f'Cat 2 (cblas in src, not exported)       : {len(cat2_cblas_unexported):4d}')
print(f'Cat 3 (*_ref in src, need wrap+export)   : {len(cat3_ref_in_source):4d}')
print(f'Cat 4 (plain impl, need cblas wrap)      : {len(cat4_plain_in_source):4d}')
print(f'Cat 5 (truly absent)                     : {len(cat5_absent):4d}')
print(f'Total actionable (wire-able)             : {total_w:4d}')
print(f'Saved: {out}')
