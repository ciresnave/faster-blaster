"""Patch reference.c to wire spectral vtable fields."""

path = r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\src\backends\reference.c'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replacements: NULL -> actual function pointers for the spectral fields we added
replacements = [
    (
        'vt.ssyev  = NULL;  vt.dsyev  = NULL;  vt.cheev  = NULL;  vt.zheev  = NULL;',
        'vt.ssyev  = (fb_ssyev_fn)ref_ssyev;  vt.dsyev  = (fb_dsyev_fn)ref_dsyev;\n'
        '    vt.cheev  = (fb_cheev_fn)ref_cheev;  vt.zheev  = (fb_zheev_fn)ref_zheev;'
    ),
    (
        'vt.sgesvd = NULL;  vt.dgesvd = NULL;  vt.cgesvd = NULL;  vt.zgesvd = NULL;',
        'vt.sgesvd = (fb_sgesvd_fn)ref_sgesvd;  vt.dgesvd = (fb_dgesvd_fn)ref_dgesvd;\n'
        '    vt.cgesvd = (fb_cgesvd_fn)ref_cgesvd;  vt.zgesvd = (fb_zgesvd_fn)ref_zgesvd;'
    ),
    (
        'vt.sgeev  = NULL;  vt.dgeev  = NULL;  vt.cgeev  = NULL;  vt.zgeev  = NULL;',
        'vt.sgeev  = (fb_sgeev_fn)ref_sgeev;  vt.dgeev  = (fb_dgeev_fn)ref_dgeev;\n'
        '    vt.cgeev  = (fb_cgeev_fn)ref_cgeev;  vt.zgeev  = (fb_zgeev_fn)ref_zgeev;'
    ),
]

# These may be on one long line that wraps; handle carefully
gesdd_old = 'vt.sgesdd  = NULL;  vt.dgesdd  = NULL;  vt.cgesdd  = NULL;  vt.zgesdd  = NULL;'
gesdd_new = ('vt.sgesdd  = (fb_sgesdd_fn)ref_sgesdd;  vt.dgesdd  = (fb_dgesdd_fn)ref_dgesdd;\n'
             '    vt.cgesdd  = NULL;  vt.zgesdd  = NULL;  /* cgesdd_ref / zgesdd_ref not yet wired */')

sygv_old = 'vt.ssygv   = NULL;  vt.dsygv   = NULL;  vt.chegv   = NULL;  vt.zhegv   = NULL;'
sygv_new = ('vt.ssygv   = (fb_ssygv_fn)ref_ssygv;  vt.dsygv   = (fb_dsygv_fn)ref_dsygv;\n'
             '    vt.chegv   = NULL;  vt.zhegv   = NULL;  /* chegv_ref / zhegv_ref not yet wired */')

replacements += [(gesdd_old, gesdd_new), (sygv_old, sygv_new)]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        count += 1
    else:
        # Try collapsing whitespace differences
        print(f'WARNING: did not find: {old[:60]!r}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Made {count}/{len(replacements)} replacements')
