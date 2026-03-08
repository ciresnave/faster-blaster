"""Analyze vtable map stems and compare to DLL exports."""
import re, os

vtmap = r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\src\core\op_vtable_map.c'
with open(vtmap, encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find k_op_stem_map section specifically (not the offset map)
stem_start = content.find('static const fb_stem_entry_t k_op_stem_map')
stem_end = content.find('};', stem_start) + 2
stem_section = content[stem_start:stem_end]
stems = re.findall(r'"([^"]+)"', stem_section)
print(f'Total stems in vtable map: {len(stems)}')

# Categorize
cats = {}
for s in stems:
    if re.match(r'^dnn_|^conv|^pool|^batch_norm|^layer_norm|^group_norm|^instance_norm|^fc_|^dropout|^softmax_|^relu_', s): c = 'dnn'
    elif re.match(r'^fft|^ifft|^rfft|^stfft', s): c = 'fft'
    elif re.match(r'^sparse_|^spmv|^spmm|^sptrsv|^mkl_sparse', s): c = 'sparse'
    elif re.match(r'^vd_|^vs_|^vz_|^vc_|^vml_', s): c = 'vml'
    elif re.match(r'^p[a-z]', s): c = 'scalapack'
    elif re.match(r'^mkl_', s): c = 'mkl'
    elif re.match(r'^nccl|^reduce_|^broadcast|^allreduce|^scatter|^gather|^prim_', s): c = 'parallel'
    elif re.match(r'^rng|^vsl_|^virnguniform', s): c = 'rng'
    elif re.match(r'^tensor|^fb_tensor', s): c = 'tensor'
    elif re.match(r'^gemm_unified|^norm_unified|^reduction_unified|^lpgemm|^geodl|^chem_', s): c = 'extended'
    else: c = 'blas_lapack_other'
    cats[c] = cats.get(c, 0) + 1

for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f'  {k:30s}: {v}')

# Now load DLL exports and see how many match
print()
import os
exp_path = os.path.join(os.environ['TEMP'], 'exports2.txt')
with open(exp_path, encoding='utf-8', errors='replace') as f:
    exp_lines = f.readlines()

dll_exports = []
for line in exp_lines:
    m = re.match(r'^\s+\d+\s+0x[0-9a-fA-F]+\s+(\S+)', line)
    if m:
        sym = m.group(1)
        # Determine stem
        if sym.startswith('cblas_'):
            stem = sym[6:]
        elif sym.startswith('fb_'):
            stem = sym[3:]
        elif sym.endswith('_') and not sym.endswith('_ref'):
            stem = sym[:-1]
        else:
            continue
        dll_exports.append(stem)

dll_stem_set = set(dll_exports)
vtmap_stem_set = set(stems)
wired = vtmap_stem_set & dll_stem_set
not_wired = vtmap_stem_set - dll_stem_set

print(f'DLL exports (stems): {len(dll_stem_set)}')
print(f'Vtable stems matched by DLL: {len(wired)}')
print(f'Vtable stems NOT in DLL: {len(not_wired)}')

# Show what categories are not wired
cats_not_wired = {}
for s in not_wired:
    if re.match(r'^dnn_|^conv|^pool|^batch_norm|^layer_norm|^relu_', s): c = 'dnn'
    elif re.match(r'^fft|^ifft|^rfft', s): c = 'fft'
    elif re.match(r'^sparse_|^spmv', s): c = 'sparse'
    elif re.match(r'^vd_|^vs_|^vz_|^vc_', s): c = 'vml'
    elif re.match(r'^p[a-z]', s): c = 'scalapack'
    elif re.match(r'^mkl_', s): c = 'mkl'
    elif re.match(r'^nccl|^reduce_|^allreduce|^broadcast', s): c = 'parallel'
    else: c = 'blas_lapack_other'
    cats_not_wired[c] = cats_not_wired.get(c, 0) + 1

print('\nNot wired by category:')
for k, v in sorted(cats_not_wired.items(), key=lambda x: -x[1]):
    print(f'  {k:30s}: {v}')
