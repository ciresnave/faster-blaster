#!/usr/bin/env python3
import re
import os

# Read the superset file
with open('BLAS_LAPACK_SUPERSET.md', 'r') as f:
    content = f.read()

# Find just the LAPACK COMPUTATIONAL section (not including other LAPACK sections)
comp_start = content.find('### LAPACK COMPUTATIONAL ROUTINES')
if comp_start == -1:
    print("Could not find section start")
    exit(1)
    
# Find the next ### which marks the end
comp_end = content.find('### LAPACK DRIVERS', comp_start)
if comp_end == -1:
    comp_end = content.find('### LAPACK', comp_start + 100)
if comp_end == -1:
    # If no next section, look for COMBINED SUPERSET
    comp_end = content.find('## COMBINED SUPERSET', comp_start)
if comp_end == -1:
    comp_end = len(content)

comp_section = content[comp_start:comp_end]
print(f"Found LAPACK COMPUTATIONAL section: {len(comp_section)} characters")

# Extract all operation names
pattern = r'`([SCDZ][A-Z0-9_]*)`'
operations = re.findall(pattern, comp_section)

# Remove duplicates and sort
operations = sorted(set(operations))

print(f"\nTotal LAPACK Computational operations: {len(operations)}")
print("\nFirst 20 operations:")
for op in operations[:20]:
    print(f"  {op}")
    
print(f"\nLast 20 operations:")
for op in operations[-20:]:
    print(f"  {op}")

# Now check what files exist in blas-lapack-reference
ref_comp_dir = r"C:\Users\cires\OneDrive\Documents\projects\blas-lapack-reference\src\lapack\computational"
if os.path.exists(ref_comp_dir):
    existing_files = set()
    for filename in os.listdir(ref_comp_dir):
        if filename.endswith('.c'):
            # Extract operation name (remove .c extension)
            op_name = filename[:-2].upper()
            existing_files.add(op_name)
    
    print(f"\n\nExisting files in blas-lapack-reference: {len(existing_files)}")
    
    # Find missing operations
    missing = [op for op in operations if op.lower() not in [f.lower() for f in existing_files]]
    print(f"Missing operations: {len(missing)}")
    
    if missing:
        print(f"\nFirst 20 missing operations:")
        for op in sorted(missing)[:20]:
            print(f"  {op}")
        
        print(f"\nLast 20 missing operations:")
        for op in sorted(missing)[-20:]:
            print(f"  {op}")
        
        # Write missing to file
        with open('lapack_comp_missing.txt', 'w') as f:
            for op in sorted(missing):
                f.write(op + '\n')
        print(f"\n✅ Wrote {len(missing)} missing operations to lapack_comp_missing.txt")
else:
    print(f"Directory not found: {ref_comp_dir}")

# Write all expected to file
with open('lapack_comp_operations_expected.txt', 'w') as f:
    for op in operations:
        f.write(op.lower() + '\n')

print(f"✅ Wrote {len(operations)} expected operations to lapack_comp_operations_expected.txt")
