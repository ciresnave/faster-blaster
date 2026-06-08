"""
gen_cblas_wrappers.py — Generate FBR_EXPORT cblas_* wrappers for all Cat 3 & Cat 4 gap items.

Strategy:
  1. Read gap_detail.txt for Cat 3 and Cat 4 items.
  2. For each source file, parse ALL function definitions → build a signature map.
  3. For each stem, find its implementation (plain, fortran_, or _ref), extract signature.
  4. Generate a thin cblas_<stem> wrapper.
  5. Write wrappers grouped into the correct destination .c files.

Output files:
  - src/blas/level2/blas_l2_cblas_wrappers.c        (appended)
  - src/blas/level3/blas_ext_cblas_wrappers.c        (new)
  - src/lapack/computational/lapack_computational_cblas_wrappers.c  (appended)
  - src/lapack/drivers/lapack_drivers_cblas_wrappers.c              (appended)
  - src/lapack/auxiliary/lapack_auxiliary_cblas_wrappers.c           (new)
"""

import re, os, sys, collections

REF_ROOT = r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster-reference'
FB_ROOT  = r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster'
GAP_FILE = os.path.join(FB_ROOT, 'gap_detail.txt')

# ── Regex helpers ──────────────────────────────────────────────────────────────

# Match a function definition (not just declaration/prototype).
# Capture: (return_type_and_decorators, function_name, params_raw)
# We'll do a multi-line scan below; this just locates candidates.
FN_HEAD_RE = re.compile(
    r'^[ \t]*(?:(?:LAPACK_KERNEL|LAPACK_AUX|LAPACK_DRIVER|BLAS_L\w*|FBR_EXPORT|static inline|inline|static|extern)\s+)*'
    r'(?:void|int|float|double|long double|float _Complex|double _Complex|size_t|unsigned int)\s+'
    r'(?:FBR_EXPORT\s+)?(\w+)\s*\(',
    re.MULTILINE
)


def extract_all_signatures(filepath):
    """
    Returns dict: function_name -> (return_type, params_string, is_definition)
    Only keeps actual definitions (has a '{' before the next function head or EOF).
    """
    try:
        with open(filepath, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception:
        return {}

    results = {}
    for m in FN_HEAD_RE.finditer(text):
        fname = m.group(1)
        if fname in ('if', 'for', 'while', 'switch', 'return', 'sizeof', 'typedef'):
            continue
        start = m.start()
        # Find the full signature: from start to the '{' or ';'
        paren_depth = 0
        pos = text.find('(', m.start())
        if pos == -1:
            continue
        paren_start = pos
        paren_depth = 0
        end_paren = pos
        for i in range(pos, min(pos + 4000, len(text))):
            c = text[i]
            if c == '(':
                paren_depth += 1
            elif c == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    end_paren = i
                    break
        # Check what follows the closing paren
        after = text[end_paren+1:end_paren+200].lstrip()
        is_definition = after.startswith('{') or after.startswith('\n{') or after.startswith('\r\n{')
        # Also accept definitions that have attributes before '{'
        if not is_definition:
            # Skip whitespace and __attribute__ etc
            stripped = re.sub(r'\s+', '', after[:50])
            is_definition = stripped.startswith('{') or stripped.startswith('__attribute')
        
        params_raw = text[paren_start+1:end_paren]
        # Determine return type
        head = text[m.start():m.start() + (pos - m.start())]
        # Strip decorators
        ret_type = re.sub(r'\b(LAPACK_KERNEL|LAPACK_AUX|LAPACK_DRIVER|BLAS_L\w*|FBR_EXPORT)\b', '', head)
        ret_type = ret_type.strip().rstrip()
        # Clean up extra whitespace
        ret_type = re.sub(r'\s+', ' ', ret_type).strip()
        # Remove the function name from end of ret_type if it leaked in
        ret_type = re.sub(r'\s+\w+\s*$', '', ret_type).strip()

        if fname not in results or is_definition:
            results[fname] = {
                'name': fname,
                'ret': ret_type,
                'params': params_raw.strip(),
                'is_def': is_definition,
            }
    return results


def clean_params(params_raw):
    """Clean up multi-line parameter string."""
    p = re.sub(r'\s+', ' ', params_raw).strip()
    # Remove trailing comma or whitespace
    p = p.rstrip(',').strip()
    return p


def parse_params(params_str):
    """
    Parse parameter string into list of (type, name) pairs.
    Returns list of param dicts: {'type': ..., 'name': ..., 'raw': ...}
    """
    if not params_str or params_str.strip() in ('void', ''):
        return []
    # Split by comma (careful with pointer decls)
    # Simple split: works for most cases
    parts = []
    depth = 0
    cur = ''
    for ch in params_str:
        if ch in '(<':
            depth += 1
        elif ch in ')>':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    
    result = []
    for p in parts:
        p = p.strip()
        if not p or p == '...':
            result.append({'type': p, 'name': '', 'raw': p})
            continue
        # Find the parameter name: last word-like token
        # Handle cases like: const float *x, int n, char uplo
        # Remove array dimensions like [4]
        p_clean = re.sub(r'\[.*?\]', '', p).strip()
        # Match: type* name or type *name or type name
        m = re.match(r'^(.*?)(\b\w+)\s*$', p_clean)
        if m:
            type_part = m.group(1).rstrip('*').strip()
            if '*' in m.group(1):
                type_part = m.group(1).strip()
            name_part = m.group(2)
        else:
            type_part = p_clean
            name_part = '_param'
        result.append({'type': type_part, 'name': name_part, 'raw': p})
    return result


# ── Fortran pointer → value conversion ────────────────────────────────────────

# For Fortran-style _ functions: all params are pointers, we convert to value-based CBLAS signatures.
# Map from pointer type to value type for scalars.
# Known integer scalar parameter names in Fortran BLAS/LAPACK (passed as const int *)
INT_SCALAR_NAMES = re.compile(
    r'^(n|m|k|lda|ldb|ldc|ldaf|ldab|ldafb|ldvl|ldvr|ldu|ldvt|ldz|ldq|ldp'
    r'|incx|incy|incz|kl|ku|kd|kd1|ks|p|q|ilo|ihi|il|iu|nrhs|nz|mm|nn|kk'
    r'|mb|nb|ib|jb|kb|mc|nc|ia|ib2|ja|jb2|lwork|liwork|lrwork|lzwork|ldwork'
    r'|j1|j2|ldh|ldvs|ldl|ldt|ldbq|ldbz|nb1|nb2|nsplit|m1|m2|n1|n2'
    r'|k1|k2|kbot|ktop|j|i|l|jcol|jrow|jlen|itmax|nmin'
    r'|batch_count|batch_size|stride_a|stride_b|stride_c|stride_x|stride_y'
    r'|ldda|lddb|lddc|mn|nfound|mfound|nz_found|nconv|ncols|nrows'
    r'|wantq|wantz)'
    r'[0-9]*$'
)

# Known floating-point SCALAR param names (NOT arrays) in Fortran BLAS/LAPACK
FLOAT_SCALAR_NAMES = re.compile(r'^(alpha|beta|anorm|bnorm|cnorm|rcond|abstol|tol|safmin|ssfmin|sfmin|eps|atol|rtol|scale1|scale2|shift|sigma)$')

# Known char (single character) parameter names
CHAR_PARAMS = re.compile(
    r'^(uplo|trans|transa|transb|diag|side|jobz|jobu|jobvt|jobvl|jobvr'
    r'|job|range|howmny|vect|balanc|sense|compz|compq|norm|storev|direct'
    r'|order|type|sort|eigsrc|initv|fact|equed|cmach|itype|jobq|jobp|method)'
    r'$'
)


def is_array_param(name, ptype):
    """Heuristic: is this parameter an array (vs. scalar)?"""
    if '*' in ptype:
        return True
    return False


def generate_fortran_cblas_wrapper(stem, sig, include_layout=False):
    """
    Generate a cblas_* wrapper for a Fortran-style function (all-pointer args).
    Produces a proper CBLAS-style wrapper converting value args to pointer.
    
    Conversion rules:
    - const int *name   → int name  (if name matches INT_SCALAR_NAMES)
    - const float/double *name → float/double name  (ONLY if name is alpha/beta/known scalar)
    - const char *name  → char name (if name matches CHAR_PARAMS)
    - everything else   → kept as-is (array pointer)
    """
    params_raw = clean_params(sig['params'])
    params = parse_params(params_raw)
    
    if not params:
        # No-arg stub
        cblas_params = 'void'
        call_args = ''
        local_decls = ''
    else:
        cblas_param_list = []
        local_decls_list = []
        call_arg_list = []
        
        for p in params:
            raw = p['raw'].strip()
            name = p['name']
            
            # Try to match: const int * name (integer scalar)
            int_scalar = re.match(
                r'^(?:const\s+)?int\s*\*\s*(\w+)$', raw.strip()
            )
            # Try to match: const float * name or const double * name (float scalar - only alpha/beta etc.)
            float_scalar = re.match(
                r'^(?:const\s+)?(float|double)\s*\*\s*(\w+)$', raw.strip()
            )
            # Try to match: const char * name (single character like uplo, trans, ...)
            char_scalar = re.match(
                r'^(?:const\s+)?char\s*\*\s*(\w+)$', raw.strip()
            )
            
            if int_scalar:
                vname = int_scalar.group(1)
                if INT_SCALAR_NAMES.match(vname):
                    cblas_param_list.append(f'int {vname}')
                    local_decls_list.append(f'    int _{vname} = {vname};')
                    call_arg_list.append(f'&_{vname}')
                else:
                    # Unknown int * — could be an array (ipiv, jpvt, etc.) — keep as pointer
                    cblas_param_list.append(raw)
                    call_arg_list.append(name)
            elif float_scalar:
                ftype = float_scalar.group(1)
                vname = float_scalar.group(2)
                if FLOAT_SCALAR_NAMES.match(vname):
                    cblas_param_list.append(f'{ftype} {vname}')
                    local_decls_list.append(f'    {ftype} _{vname} = {vname};')
                    call_arg_list.append(f'&_{vname}')
                else:
                    # Unknown float * — almost certainly an array — keep as pointer
                    cblas_param_list.append(raw)
                    call_arg_list.append(name)
            elif char_scalar:
                vname = char_scalar.group(1)
                if CHAR_PARAMS.match(vname):
                    cblas_param_list.append(f'char {vname}')
                    local_decls_list.append(f'    char _{vname} = {vname};')
                    call_arg_list.append(f'&_{vname}')
                else:
                    cblas_param_list.append(raw)
                    call_arg_list.append(name)
            else:
                # Complex pointers, array pointers, etc. — keep as-is
                cblas_param_list.append(raw)
                call_arg_list.append(name)
        
        cblas_params = ', '.join(cblas_param_list)
        local_decls = '\n'.join(local_decls_list)
        call_args = ', '.join(call_arg_list)
    
    ret = sig['ret'] if sig['ret'] else 'void'
    fn_name = sig['name']
    cblas_name = f'cblas_{stem}'
    
    lines = []
    lines.append(f'/* Forward declaration */')
    lines.append(f'extern {ret} {fn_name}({params_raw});')
    if local_decls:
        lines.append(f'FBR_EXPORT {ret} {cblas_name}({cblas_params}) {{')
        lines.append(local_decls)
        if ret != 'void':
            lines.append(f'    return {fn_name}({call_args});')
        else:
            lines.append(f'    {fn_name}({call_args});')
        lines.append('}')
    else:
        if ret != 'void':
            lines.append(f'FBR_EXPORT {ret} {cblas_name}({cblas_params or "void"}) {{')
            lines.append(f'    return {fn_name}({call_args});')
            lines.append('}')
        else:
            lines.append(f'FBR_EXPORT {ret} {cblas_name}({cblas_params or "void"}) {{')
            lines.append(f'    {fn_name}({call_args});')
            lines.append('}')
    return '\n'.join(lines)


def is_all_pointer_params(params_raw):
    """
    Heuristic: returns True if this looks like a Fortran-alias function
    (i.e., all non-void params are by pointer).
    """
    if not params_raw or params_raw.strip() in ('void', ''):
        return False  # no-arg: treat as plain
    params = parse_params(params_raw)
    if len(params) == 1 and 'info' in params[0]['name']:
        return True  # single int *info stub - treat as plain pass-through
    pointer_count = sum(1 for p in params if '*' in p['raw'])
    total = len([p for p in params if p['raw'] and p['raw'] != '...'])
    # If > 60% of params are pointers, likely Fortran-alias
    return total > 1 and (pointer_count / total) >= 0.6


def generate_plain_cblas_wrapper(stem, sig):
    """
    Generate a cblas_* wrapper for a plain C-style function (no pointer conversion needed).
    Just forward all params as-is.
    """
    params_raw = clean_params(sig['params'])
    ret = sig['ret'] if sig['ret'] else 'void'
    fn_name = sig['name']
    cblas_name = f'cblas_{stem}'
    
    params = parse_params(params_raw)
    arg_names = [p['name'] for p in params if p['name'] and p['name'] != '...']
    call_args = ', '.join(arg_names)
    
    lines = []
    lines.append(f'/* Forward declaration */')
    lines.append(f'extern {ret} {fn_name}({params_raw if params_raw else "void"});')
    if ret != 'void':
        lines.append(f'FBR_EXPORT {ret} {cblas_name}({params_raw if params_raw else "void"}) {{')
        lines.append(f'    return {fn_name}({call_args});')
    else:
        lines.append(f'FBR_EXPORT void {cblas_name}({params_raw if params_raw else "void"}) {{')
        lines.append(f'    {fn_name}({call_args});')
    lines.append('}')
    return '\n'.join(lines)


# ── Destination file mapping ───────────────────────────────────────────────────

def get_dest_file(source_path):
    """Map a source file path to its wrapper output file."""
    sp = source_path.replace('\\', '/')
    if 'blas/level2' in sp:
        return 'blas_l2'
    if 'blas/level3' in sp or 'blas/level' in sp:
        return 'blas_l3_ext'
    if 'lapack/computational' in sp:
        return 'lapack_comp'
    if 'lapack/drivers' in sp:
        return 'lapack_drivers'
    if 'lapack/auxiliary' in sp:
        return 'lapack_aux'
    if 'lapack/' in sp:
        return 'lapack_aux'
    return 'misc'


DEST_FILE_PATHS = {
    'blas_l2':       os.path.join(REF_ROOT, 'src', 'blas', 'level2', 'blas_l2_cblas_wrappers.c'),
    'blas_l3_ext':   os.path.join(REF_ROOT, 'src', 'blas', 'level3', 'blas_ext_cblas_wrappers.c'),
    'lapack_comp':   os.path.join(REF_ROOT, 'src', 'lapack', 'computational', 'lapack_computational_cblas_wrappers.c'),
    'lapack_drivers':os.path.join(REF_ROOT, 'src', 'lapack', 'drivers', 'lapack_drivers_cblas_wrappers.c'),
    'lapack_aux':    os.path.join(REF_ROOT, 'src', 'lapack', 'auxiliary', 'lapack_auxiliary_cblas_wrappers.c'),
    'misc':          os.path.join(REF_ROOT, 'src', 'lapack', 'auxiliary', 'lapack_auxiliary_cblas_wrappers.c'),
}

NEW_FILE_HEADERS = {
    'blas_l3_ext': """\
/* Auto-generated BLAS extension (MKL-compat) cblas_* wrappers */
#include <blas_reference.h>
#include "../../include/fbr_export.h"
#include <stddef.h>
#include <stdint.h>

""",
    'lapack_aux': """\
/* Auto-generated LAPACK auxiliary cblas_* wrappers */
#include <lapack_reference.h>
#include "../../include/fbr_export.h"
#include <stddef.h>

""",
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Read gap_detail.txt
    items = []  # list of {'cat', 'stem', 'symbol', 'src_file_rel'}
    with open(GAP_FILE, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            cat, stem, symbol, src_file_rel = parts[0], parts[1], parts[2], parts[3]
            if cat not in ('3_REF_IN_SOURCE', '4_PLAIN_IN_SOURCE'):
                continue
            items.append({'cat': cat, 'stem': stem, 'symbol': symbol, 'src': src_file_rel})

    print(f'Items to process: {len(items)}')

    # 2. Group items by source file (to read each file once)
    by_file = collections.defaultdict(list)
    for it in items:
        by_file[it['src']].append(it)

    # 3. For each source file, extract signatures
    sig_map = {}  # stem -> sig dict
    skipped = []
    
    for src_rel, its in by_file.items():
        src_abs = os.path.join(REF_ROOT, src_rel.replace('/', os.sep))
        if not os.path.exists(src_abs):
            print(f'  [MISSING] {src_rel}')
            for it in its:
                skipped.append(it['stem'])
            continue
        
        sigs = extract_all_signatures(src_abs)
        
        for it in its:
            stem = it['stem']
            symbol = it['symbol']
            
            # Try in order: exact symbol, symbol_, symbol_ref, stem, stem_, stem_ref
            candidates = [symbol, symbol + '_', symbol + '_ref', stem, stem + '_', stem + '_ref']
            found = None
            for cand in candidates:
                if cand in sigs:
                    found = sigs[cand]
                    break
            
            # Also try looking in the COMPUTATIONAL directory for Cat 3 items from drivers
            if found is None and it['cat'] == '3_REF_IN_SOURCE':
                # Try to find in computational files
                comp_candidates = [
                    os.path.join(REF_ROOT, 'src', 'lapack', 'computational', f'{stem}.c'),
                    os.path.join(REF_ROOT, 'src', 'lapack', 'computational', f'{stem.lower()}.c'),
                ]
                for comp_file in comp_candidates:
                    if os.path.exists(comp_file):
                        comp_sigs = extract_all_signatures(comp_file)
                        for cand in [stem, stem + '_', stem + '_ref']:
                            if cand in comp_sigs:
                                found = comp_sigs[cand]
                                it['src'] = f'src/lapack/computational/{os.path.basename(comp_file)}'
                                break
                    if found:
                        break
            
            if found is None:
                print(f'  [NO SIG] stem={stem} symbol={symbol} in {src_rel}')
                skipped.append(stem)
                continue
            
            sig_map[stem] = {'sig': found, 'src': it['src']}

    print(f'Signatures found: {len(sig_map)}, skipped: {len(skipped)}')

    # 4. Generate wrappers grouped by destination
    dest_wrappers = collections.defaultdict(list)  # dest_key -> list of wrapper strings
    
    for stem, info in sorted(sig_map.items()):
        sig = info['sig']
        src = info['src']
        dest = get_dest_file(src)
        
        fn_name = sig['name']
        params_raw = clean_params(sig['params'])
        is_fortran = fn_name.endswith('_') and not fn_name.endswith('__')
        is_ref    = fn_name.endswith('_ref')
        
        if is_fortran:
            wrapper = generate_fortran_cblas_wrapper(stem, sig)
        elif not is_ref and is_all_pointer_params(params_raw):
            # Plain C function that's really a Fortran alias (all-pointer convention)
            # Apply the same scalar-value conversions
            wrapper = generate_fortran_cblas_wrapper(stem, sig)
        else:
            wrapper = generate_plain_cblas_wrapper(stem, sig)
        
        dest_wrappers[dest].append(f'/* {stem} */\n{wrapper}')

    # 5. Write output
    already_exported = set()
    
    for dest_key, wrappers in dest_wrappers.items():
        out_path = DEST_FILE_PATHS[dest_key]
        
        # Check which cblas_ functions already exist in the output file
        if os.path.exists(out_path):
            with open(out_path, encoding='utf-8', errors='replace') as f:
                existing = f.read()
            mode = 'a'  # append
        else:
            existing = ''
            mode = 'w'  # create new

        # Filter out already-present wrappers
        new_wrappers = []
        for w in wrappers:
            # Extract the cblas name from the wrapper
            m = re.search(r'FBR_EXPORT \w+ (cblas_\w+)\s*\(', w)
            if m:
                cblas_name = m.group(1)
                if cblas_name in existing or cblas_name in already_exported:
                    continue
                already_exported.add(cblas_name)
            new_wrappers.append(w)
        
        if not new_wrappers:
            print(f'  [SKIP] {os.path.basename(out_path)} — all wrappers already present')
            continue
        
        print(f'  Writing {len(new_wrappers)} wrappers to {os.path.basename(out_path)} ({mode})')
        
        with open(out_path, mode=mode, encoding='utf-8') as f:
            if mode == 'w':
                header = NEW_FILE_HEADERS.get(dest_key, '/* Auto-generated cblas_* wrappers */\n#include <lapack_reference.h>\n#include "../../include/fbr_export.h"\n\n')
                f.write(header)
            else:
                f.write('\n\n/* ── Auto-generated wrappers (Cat3/Cat4 gap fill) ── */\n\n')
            
            for w in new_wrappers:
                f.write(w)
                f.write('\n\n')
    
    # Report skipped
    if skipped:
        print(f'\nSkipped {len(skipped)} stems (no signature found):')
        for s in sorted(skipped)[:30]:
            print(f'  {s}')
        if len(skipped) > 30:
            print(f'  ... and {len(skipped)-30} more')
    
    print('\nDone.')


if __name__ == '__main__':
    main()
