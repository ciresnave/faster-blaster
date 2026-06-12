#!/usr/bin/env python3
import re

# Read the superset file
with open('BLAS_LAPACK_SUPERSET.md', 'r') as f:
    content = f.read()

# Find the LAPACK COMPUTATIONAL section
comp_section = re.search(r'### LAPACK COMPUTATIONAL ROUTINES.*?(?=###|$)', content, re.DOTALL)
if not comp_section:
    print("Could not find LAPACK COMPUTATIONAL section")
    exit(1)

# Extract all operation names (S, D, C, Z variants, uppercase)
pattern = r'`([SCDZ][A-Z0-9_]+)`'
operations = re.findall(pattern, comp_section.group(0))
# Filter only those that start with uppercase letters following SCDZ
operations = [op for op in operations if len(op) > 1]

# Remove duplicates and sort
operations = sorted(set(operations))

print(f"Total LAPACK Computational operations in superset: {len(operations)}")
print("\nFirst 20 operations:")
for op in operations[:20]:
    print(f"  {op}")
    
print(f"\nLast 20 operations:")
for op in operations[-20:]:
    print(f"  {op}")

# Write to file for later use
with open('lapack_comp_operations_expected.txt', 'w') as f:
    for op in operations:
        f.write(op.lower() + '\n')

print(f"\n✅ Wrote {len(operations)} operations to lapack_comp_operations_expected.txt")
