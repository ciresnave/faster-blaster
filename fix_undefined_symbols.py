#!/usr/bin/env python3
"""Fix all undefined symbols across all lapack *_cblas_wrappers.c files."""
import re, sys, os

BASE = r"c:\Users\cires\OneDrive\Documents\projects\faster-blaster-reference\src\lapack"
WRAPPER_FILES = [
    os.path.join(BASE, "drivers", "lapack_drivers_cblas_wrappers.c"),
    os.path.join(BASE, "computational", "lapack_computational_cblas_wrappers.c"),
    os.path.join(BASE, "auxiliary", "lapack_auxiliary_cblas_wrappers.c"),
]

UNDEFINED = set(sys.argv[1:]) if len(sys.argv) > 1 else {
    'cgbsvx','cgeesx','cgelsd','cgesvxx','cgges','cggesx','cggev','cggevx',
    'cggglm','cgglse','cgtsv','cgtsvx','cheevr','chesvxx','chseqr_ref',
    'cpbsvx','cposvxx','cppsvx','csysvxx','ctrevc_ref',
    'dgelsd','dgges','dggglm','dgglse','dhseqr_ref','dorglq',
    'dpbsvx','dposvxx','dppsvx','dsbevd','dsbevx','dsyevr','dsysvx','dsysvxx','dtrevc_ref',
    'sgelsd','sgges','sggglm','sgglse','shseqr_ref',
}

def get_cblas_name(u): return 'cblas_' + u.replace('_ref','')
def make_stub(sig, u):
    is_int = bool(re.search(r'\bFBR_EXPORT\s+int\b', sig))
    has_info = 'int *info' in sig
    if is_int: return '    /* stub — underlying %s not compiled */\n    return -999;\n' % u
    if has_info: return '    /* stub — underlying %s not compiled */\n    if (info) *info = -999;\n' % u
    return '    /* stub — underlying %s not compiled */\n    return;\n' % u

def fix_file(path, undefs):
    if not os.path.exists(path): return 0
    src = open(path,'r',encoding='utf-8').read(); orig = src; n = 0
    for u in sorted(undefs):
        cblas = get_cblas_name(u)
        if cblas not in src: continue
        src = re.sub(r'extern\s+(?:void|int)\s+'+re.escape(u)+r'\s*\([^;]*?\)\s*;[ \t]*\n','',src,flags=re.DOTALL)
        m = re.search(r'(FBR_EXPORT\s+(?:void|int)\s+'+re.escape(cblas)+r'\s*\([^{]*\)\s*\{)',src,re.DOTALL)
        if not m: continue
        i, depth = m.end(), 1
        while i < len(src) and depth:
            if src[i]=='{': depth+=1
            elif src[i]=='}': depth-=1
            i+=1
        body = src[m.end():i]
        if 'stub' in body: continue
        src = src[:m.start()] + m.group(1) + '\n' + make_stub(m.group(1),u) + '}\n' + src[i:]
        n+=1; print(f'  [{os.path.basename(path)}] Stubbed {cblas}')
    if src != orig: open(path,'w',encoding='utf-8',newline='\n').write(src)
    return n

total = sum(fix_file(f,UNDEFINED) for f in WRAPPER_FILES)
print(f'\nTotal: {total} wrappers stubbed.')
