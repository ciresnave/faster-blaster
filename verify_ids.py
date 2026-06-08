import re
lines = open('src/judge/judge_op_ids.h', encoding='utf-8').readlines()
sym_to_id = {}
for l in lines:
    m = re.match(r'#define (FB_OP_\w+)\s+(\d+)', l.strip())
    if m:
        sym_to_id[m.group(1)] = int(m.group(2))

families = [
    ['FB_OP_SGETRF','FB_OP_DGETRF','FB_OP_CGETRF','FB_OP_ZGETRF'],
    ['FB_OP_SGETRI','FB_OP_DGETRI','FB_OP_CGETRI','FB_OP_ZGETRI'],
    ['FB_OP_SGESV', 'FB_OP_DGESV', 'FB_OP_CGESV', 'FB_OP_ZGESV'],
    ['FB_OP_SAXPBY','FB_OP_DAXPBY','FB_OP_CAXPBY','FB_OP_ZAXPBY'],
    ['FB_OP_SBDSDC','FB_OP_DBDSDC'],
    ['FB_OP_SGBMV','FB_OP_DGBMV','FB_OP_CGBMV','FB_OP_ZGBMV'],
]
for fam in families:
    ids = [sym_to_id.get(s,'???') for s in fam]
    ok = all(isinstance(ids[i],int) and isinstance(ids[i+1],int) and ids[i+1]==ids[i]+1 for i in range(len(ids)-1))
    tag = "OK" if ok else "FAIL"
    print("  " + tag + " " + fam[0] + ": " + str(ids))

print()
print("  FB_OP_ZGEEV = " + str(sym_to_id.get('FB_OP_ZGEEV')) + " (expect 313)")
print("  FB_OP_SAXPY = " + str(sym_to_id.get('FB_OP_SAXPY')) + " (expect 0)")
print("  Max ID: " + str(max(sym_to_id.values())))
print("  Total defines: " + str(len(sym_to_id)))
