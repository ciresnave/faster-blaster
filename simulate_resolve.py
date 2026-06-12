import re
import sys

def parse_stem_map(filepath):
    stem_map = []
    pattern = re.compile(r'\{\s*"([^"]+)"\s*,\s*(FB_OP_[A-Z0-9_]+)\s*\}')
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        start_match = re.search(r'static const fb_stem_entry_t k_op_stem_map\[\] = \{', content)
        if not start_match: return []
        start = start_match.end()
        end = content.find('};', start)
        array_content = content[start:end]
        for match in pattern.finditer(array_content):
            stem_map.append((match.group(1), match.group(2)))
    return stem_map

def parse_exports(filepath):
    exports = []
    pattern = re.compile(r'Name: (.*)')
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                exports.append(match.group(1).strip())
    return exports

def fb_classify_symbol(name, stems_dict):
    if name in stems_dict:
        return stems_dict[name]
    prefixes = ["cblas_", "fb_", "LAPACKE_"]
    for p in prefixes:
        if name.startswith(p):
            stem = name[len(p):]
            if stem in stems_dict:
                return stems_dict[stem]
    if name.endswith("_"):
        stem = name[:-1]
        if stem in stems_dict:
            return stems_dict[stem]
    return None

def main():
    stem_map_path = r'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\src\core\op_vtable_map.c'
    exports_path = 'exports.txt'
    
    stem_map = parse_stem_map(stem_map_path)
    exports = parse_exports(exports_path)
    stems_dict = {s: op_id for s, op_id in stem_map}
    
    print(f"Total entries in k_op_stem_map: {len(stem_map)}")
    print(f"Total exports: {len(exports)}")
    
    resolved_op_ids = set()
    for exp in exports:
        op_id = fb_classify_symbol(exp, stems_dict)
        if op_id:
            resolved_op_ids.add(op_id)
            
    print(f"Unique op_ids resolved: {len(resolved_op_ids)}")
    
    check_stems = ["sasum", "saxpy", "sgeev", "sgeqlf", "sgelsy"]
    print("Resolution of specific stems:")
    for stem in check_stems:
        target_op_id = stems_dict.get(stem)
        resolved = target_op_id in resolved_op_ids
        print(f"  {stem}: {'RESOLVED' if resolved else 'NOT RESOLVED'} ({target_op_id})")
        
    all_pairs = []
    seen_op_ids = set()
    for stem, op_id in stem_map:
        if op_id not in seen_op_ids:
            all_pairs.append((stem, op_id))
            seen_op_ids.add(op_id)
            
    print(f"Total canonical op_ids in map: {len(all_pairs)}")
    unresolved = [f"{op_id} ({stem})" for stem, op_id in all_pairs if op_id not in resolved_op_ids]
    
    print(f"Total unresolved canonical op_ids: {len(unresolved)}")
    if unresolved:
        print("\nFirst 30 unresolved canonical op_ids:")
        for item in unresolved[:30]:
            print(f"  {item}")
    else:
        print("\nNo unresolved canonical op_ids found.")

if __name__ == '__main__':
    main()
