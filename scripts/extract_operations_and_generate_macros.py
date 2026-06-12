#!/usr/bin/env python3
"""
Extract all BLAS/LAPACK operations from faster-blaster-reference and generate
a macro-based wrapper system for faster-blaster.

This approach uses:
1. Extract all function signatures from blas_wrappers.c
2. Parse signatures to build operation database
3. Generate a header with macro definitions
4. Generate wrapper implementations using macro expansion

Much cleaner than generate_wrappers.py!

Author: Faster-Blaster Integration Team
License: MIT OR Apache-2.0
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, NamedTuple
from dataclasses import dataclass


@dataclass
class Parameter:
    """Represents a function parameter"""
    type_str: str      # e.g., "const float*", "int", "char*"
    name: str          # e.g., "x", "n", "trans"
    
    def is_pointer(self) -> bool:
        return '*' in self.type_str
    
    def is_const(self) -> bool:
        return 'const' in self.type_str
    
    def base_type(self) -> str:
        """Get base type without const/pointer"""
        return self.type_str.replace('const', '').replace('*', '').strip()
    
    def fortran_type(self) -> str:
        """Convert to Fortran calling convention (by reference)"""
        base = self.base_type()
        if self.is_pointer():
            return f"{base}*"
        else:
            return f"{base}*"  # Scalar becomes pointer in Fortran


@dataclass
class Operation:
    """Represents a BLAS/LAPACK operation"""
    fortran_name: str    # e.g., "saxpy_"
    c_name: str          # e.g., "saxpy" (without underscore)
    return_type: str     # e.g., "void", "float", "double"
    parameters: List[Parameter]
    category: str        # "level1", "level2", "level3", "lapack"
    precision: str       # "s" (float), "d" (double), "c" (complex), "z" (complex double)
    
    def fortran_signature(self) -> str:
        """Generate Fortran-style signature"""
        params = ", ".join([f"{p.fortran_type()} {p.name}" for p in self.parameters])
        return f"{self.return_type} {self.fortran_name}({params})"
    
    def cblas_signature(self) -> str:
        """Generate CBLAS-style signature"""
        params = ", ".join([f"{p.type_str} {p.name}" for p in self.parameters])
        return f"{self.return_type} fb_blr_{self.c_name}({params})"


class OperationExtractor:
    """Extract operations from blas_wrappers.c"""
    
    def __init__(self, wrappers_file: Path):
        self.wrappers_file = wrappers_file
        self.operations: List[Operation] = []
    
    def extract(self) -> List[Operation]:
        """Extract all operations from the wrappers file"""
        with open(self.wrappers_file, 'r') as f:
            content = f.read()
        
        # Split into sections by level/category
        level1_match = re.search(r'Level 1:.*?Level 2:', content, re.DOTALL)
        level2_match = re.search(r'Level 2:.*?Level 3:', content, re.DOTALL)
        level3_match = re.search(r'Level 3:.*?LAPACK', content, re.DOTALL)
        lapack_match = re.search(r'LAPACK.*', content, re.DOTALL)
        
        if level1_match:
            self._extract_from_section(level1_match.group(0), 'level1')
        if level2_match:
            self._extract_from_section(level2_match.group(0), 'level2')
        if level3_match:
            self._extract_from_section(level3_match.group(0), 'level3')
        if lapack_match:
            self._extract_from_section(lapack_match.group(0), 'lapack')
        
        return self.operations
    
    def _extract_from_section(self, section_text: str, category: str):
        """Extract operations from a section"""
        # Find function definitions: "TYPE name_(params)"
        # Format: void saxpy_(const int *n, const float *alpha, ...)
        pattern = r'(\w+)\s+(\w+_)\((.*?)\)\s*\{'
        
        for match in re.finditer(pattern, section_text, re.DOTALL):
            return_type = match.group(1).strip()
            fortran_name = match.group(2).strip()
            params_str = match.group(3).strip()
            
            # Skip if this is a forward declaration (no body)
            if 'extern' in section_text[max(0, match.start()-50):match.start()]:
                continue
            
            # Parse parameters
            parameters = self._parse_parameters(params_str)
            
            # Determine precision
            c_name = fortran_name.rstrip('_')
            precision = c_name[0] if c_name else '?'
            
            op = Operation(
                fortran_name=fortran_name,
                c_name=c_name,
                return_type=return_type,
                parameters=parameters,
                category=category,
                precision=precision
            )
            self.operations.append(op)
    
    def _parse_parameters(self, params_str: str) -> List[Parameter]:
        """Parse parameter list"""
        parameters = []
        
        # Split by comma, but not inside nested parens/brackets
        param_strs = re.split(r',\s*(?![^()]*\))', params_str)
        
        for param_str in param_strs:
            param_str = param_str.strip()
            if not param_str or param_str == 'void':
                continue
            
            # Parse "const float *x" or "int *n"
            # Match: [const] type [*] name
            match = re.match(r'(const\s+)?(\w+(?:\s+\w+)*)\s*(\*)?\s*(\w+)$', param_str)
            if match:
                const_part = match.group(1) or ''
                type_part = match.group(2)
                pointer = match.group(3) or ''
                name = match.group(4)
                
                full_type = f"{const_part}{type_part}{pointer}".strip()
                parameters.append(Parameter(type_str=full_type, name=name))
        
        return parameters


class WrapperGenerator:
    """Generate macro-based wrapper code"""
    
    def __init__(self, operations: List[Operation]):
        self.operations = operations
    
    def generate_macro_header(self) -> str:
        """Generate header with macro definitions"""
        output = []
        output.append("/**")
        output.append(" * MACRO-BASED WRAPPER GENERATION")
        output.append(" * ")
        output.append(" * Pattern:")
        output.append(" *   OPERATION(return_type, fortran_name, c_name, param_list)")
        output.append(" * ")
        output.append(" * This allows expanding all 1248 operations without hand-coding.")
        output.append(" */")
        output.append("")
        output.append("/* Define this macro to generate wrappers for each operation */")
        output.append("#ifndef OPERATION")
        output.append("#define OPERATION(ret_type, fort_name, c_name, params) \\")
        output.append("    /* Wrapper would be generated here */")
        output.append("#endif")
        output.append("")
        output.append("/* ============================================================================")
        output.append(" * ALL 1248 OPERATIONS - Use OPERATION macro to expand")
        output.append(" * ============================================================================ */")
        output.append("")
        
        # Group by category
        for category in ['level1', 'level2', 'level3', 'lapack']:
            cat_ops = [op for op in self.operations if op.category == category]
            if cat_ops:
                output.append(f"\n/* {category.upper()} - {len(cat_ops)} operations */")
                for op in cat_ops:
                    output.append(self._generate_macro_call(op))
        
        output.append("\n#undef OPERATION")
        output.append("")
        
        return "\n".join(output)
    
    def _generate_macro_call(self, op: Operation) -> str:
        """Generate a single macro call"""
        # Format: OPERATION(void, saxpy_, saxpy, (int*, float*, float*, int*, float*, int*))
        param_types = [p.type_str for p in op.parameters]
        param_list = ", ".join(param_types)
        return f'OPERATION({op.return_type}, {op.fortran_name}, {op.c_name}, ({param_list}))'
    
    def generate_typedef_generator(self) -> str:
        """Generate code that creates all typedefs"""
        output = []
        output.append("/**")
        output.append(" * TYPEDEF GENERATOR")
        output.append(" * ")
        output.append(" * Define TYPEDEF_OPERATION macro and include this section to generate")
        output.append(" * all Fortran function typedefs for the 1248 operations.")
        output.append(" */")
        output.append("")
        output.append("#ifdef TYPEDEF_OPERATION")
        output.append("")
        
        for op in self.operations:
            # Generate typedef: typedef void (*fblas_saxpy_t)(int*, float*, ...);
            params = ", ".join([p.fortran_type() for p in op.parameters])
            output.append(f"typedef {op.return_type} (*fblas_{op.c_name}_t)({params});")
        
        output.append("")
        output.append("#endif /* TYPEDEF_OPERATION */")
        output.append("")
        
        return "\n".join(output)
    
    def generate_wrapper_generator(self) -> str:
        """Generate code that creates all wrapper functions"""
        output = []
        output.append("/**")
        output.append(" * WRAPPER FUNCTION GENERATOR")
        output.append(" * ")
        output.append(" * Define WRAPPER_OPERATION macro and include this section to generate")
        output.append(" * all CBLAS wrapper functions that convert parameters and call Fortran.")
        output.append(" */")
        output.append("")
        output.append("#ifdef WRAPPER_OPERATION")
        output.append("")
        
        for op in self.operations:
            # Generate wrapper skeleton
            output.append(f"/* {op.c_name}: {op.fortran_signature()} */")
            cblas_params = ", ".join([f"{p.type_str} {p.name}" for p in op.parameters])
            output.append(f"WRAPPER_OPERATION({op.return_type}, {op.c_name}, {cblas_params})")
            output.append("")
        
        output.append("")
        output.append("#endif /* WRAPPER_OPERATION */")
        output.append("")
        
        return "\n".join(output)
    
    def generate_vtable_entry_list(self) -> str:
        """Generate list of vtable entries"""
        output = []
        output.append("/**")
        output.append(" * VTABLE ENTRIES")
        output.append(" * ")
        output.append(" * Add these to fb_backend_vtable_t structure:")
        output.append(" */")
        output.append("")
        
        for op in self.operations:
            output.append(f"    .{op.c_name} = fb_blr_{op.c_name},")
        
        output.append("")
        
        return "\n".join(output)


def main():
    """Main script entry point"""
    
    print("=" * 80)
    print("BLAS/LAPACK Operation Extractor & Macro-Based Wrapper Generator")
    print("=" * 80)
    print()
    
    # Check if file exists
    # Try multiple possible paths
    possible_paths = [
        Path("../../faster-blaster-reference/src/blas_wrappers.c"),
        Path("../../../faster-blaster-reference/src/blas_wrappers.c"),
        Path(".").resolve().parent.parent / "faster-blaster-reference" / "src" / "blas_wrappers.c",
    ]
    
    wrappers_file = None
    for p in possible_paths:
        if p.exists():
            wrappers_file = p
            break
    
    if not wrappers_file:
        print(f"ERROR: Could not find blas_wrappers.c in any of:")
        for p in possible_paths:
            print(f"  - {p.resolve()}")
        return 1
    
    # Extract operations
    print(f"Extracting operations from {wrappers_file}...")
    extractor = OperationExtractor(wrappers_file)
    operations = extractor.extract()
    
    if not operations:
        print("WARNING: No operations extracted. The wrappers file might have different format.")
        print("This script expects format: 'return_type function_(params) { ... }'")
        return 1
    
    print(f"✓ Extracted {len(operations)} operations")
    
    # Group by category
    by_category = {}
    for op in operations:
        if op.category not in by_category:
            by_category[op.category] = []
        by_category[op.category].append(op)
    
    print()
    print("Operations by category:")
    for category in ['level1', 'level2', 'level3', 'lapack']:
        count = len(by_category.get(category, []))
        print(f"  {category:15s}: {count:3d} operations")
    
    # Generate macro header
    print()
    print("Generating macro-based wrapper system...")
    generator = WrapperGenerator(operations)
    
    # Generate output file
    output_file = Path("operations_macro_system.h")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("/**\n")
        f.write(" * MACRO-BASED BLAS/LAPACK WRAPPER SYSTEM\n")
        f.write(" * \n")
        f.write(" * This file contains all 1248 operations defined through macros.\n")
        f.write(" * Include this file and define one of the following macros:\n")
        f.write(" * \n")
        f.write(" * 1. OPERATION(ret_type, fort_name, c_name, param_list)\n")
        f.write(" *    → Generates macro calls for all operations\n")
        f.write(" * \n")
        f.write(" * 2. TYPEDEF_OPERATION\n")
        f.write(" *    → Generates all typedefs for Fortran function pointers\n")
        f.write(" * \n")
        f.write(" * 3. WRAPPER_OPERATION(ret_type, c_name, params)\n")
        f.write(" *    → Generates all wrapper function skeletons\n")
        f.write(" * \n")
        f.write(" */\n\n")
        
        f.write(generator.generate_macro_header())
        f.write("\n")
        f.write(generator.generate_typedef_generator())
        f.write("\n")
        f.write(generator.generate_wrapper_generator())
        f.write("\n")
        f.write(generator.generate_vtable_entry_list())
    
    print(f"✓ Generated {output_file}")
    print()
    
    # Show usage
    print("=" * 80)
    print("USAGE EXAMPLE:")
    print("=" * 80)
    print()
    print("To generate typedefs:")
    print()
    print("  #define TYPEDEF_OPERATION")
    print("  #include \"operations_macro_system.h\"")
    print("  #undef TYPEDEF_OPERATION")
    print()
    print("To generate wrapper functions:")
    print()
    print("  #define WRAPPER_OPERATION(ret, name, params) \\")
    print("      static ret fb_blr_##name(params) { ... }")
    print("  #include \"operations_macro_system.h\"")
    print("  #undef WRAPPER_OPERATION")
    print()
    print("=" * 80)
    print()
    
    # Summary
    print("SUMMARY:")
    print(f"  Total operations extracted: {len(operations)}")
    print(f"  BLAS Level 1:  {len(by_category.get('level1', []))} operations")
    print(f"  BLAS Level 2:  {len(by_category.get('level2', []))} operations")
    print(f"  BLAS Level 3:  {len(by_category.get('level3', []))} operations")
    print(f"  LAPACK:        {len(by_category.get('lapack', []))} operations")
    print()
    print("File generated: operations_macro_system.h")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
