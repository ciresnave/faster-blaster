#!/usr/bin/env python3
import re

# Read the superset file
with open('BLAS_LAPACK_SUPERSET.md', 'r') as f:
    content = f.read()

# Find the LAPACK COMPUTATIONAL section
comp_start = content.find('### LAPACK COMPUTATIONAL ROUTINES')
if comp_start == -1:
    print("Could not find section start")
    exit(1)
    
comp_end = content.find('\n### ', comp_start + 1)
if comp_end == -1:
    comp_end = len(content)

comp_section = content[comp_start:comp_end]
print(f"Found LAPACK section: {len(comp_section)} characters")
print(f"First 500 chars of section:")
print(comp_section[:500])
print("\n")

# Try different regex patterns
patterns = [
    r'`([SCDZ][A-Z0-9_]+)`',
    r'[0-9]+\. `([^`]+)`',
    r'`([A-Z0-9]+)`',
]

for i, pattern in enumerate(patterns):
    matches = re.findall(pattern, comp_section)
    print(f"Pattern {i+1}: {len(matches)} matches")
    if matches:
        print(f"  First 5: {matches[:5]}")
        print(f"  Last 5: {matches[-5:]}")
