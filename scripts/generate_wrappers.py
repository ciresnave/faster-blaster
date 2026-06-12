#!/usr/bin/env python3
"""
Generate wrapper functions for all 1248 BLAS/LAPACK operations.

This script:
1. Reads the operation list from faster-blaster-reference
2. Generates Fortran-style wrapper function typedefs
3. Generates CBLAS-style wrapper implementations
4. Appends to blas_lapack_reference_backend.c

The pattern is:
- Fortran function: void saxpy_(int* n, float* alpha, ...)
- CBLAS wrapper: void fb_blr_saxpy(int n, float alpha, ...)
  which converts parameters to by-reference and calls Fortran

Author: Faster-Blaster Integration Team
License: MIT OR Apache-2.0
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


# Operation database (subset shown - full list would be loaded from operations.h)
OPERATIONS = [
    # Level 1 BLAS
    ("saxpy", "void", [("int", "n"), ("float", "alpha"), ("const float*", "x"), 
                       ("int", "incx"), ("float*", "y"), ("int", "incy")]),
    ("daxpy", "void", [("int", "n"), ("double", "alpha"), ("const double*", "x"),
                       ("int", "incx"), ("double*", "y"), ("int", "incy")]),
    ("sdot", "float", [("int", "n"), ("const float*", "x"), ("int", "incx"),
                       ("const float*", "y"), ("int", "incy")]),
    ("ddot", "double", [("int", "n"), ("const double*", "x"), ("int", "incx"),
                        ("const double*", "y"), ("int", "incy")]),
    ("scopy", "void", [("int", "n"), ("const float*", "x"), ("int", "incx"),
                       ("float*", "y"), ("int", "incy")]),
    ("dcopy", "void", [("int", "n"), ("const double*", "x"), ("int", "incx"),
                       ("double*", "y"), ("int", "incy")]),
    ("sscal", "void", [("int", "n"), ("float", "alpha"), ("float*", "x"), ("int", "incx")]),
    ("dscal", "void", [("int", "n"), ("double", "alpha"), ("double*", "x"), ("int", "incx")]),
    ("snorm2", "float", [("int", "n"), ("const float*", "x"), ("int", "incx")]),
    ("dnorm2", "double", [("int", "n"), ("const double*", "x"), ("int", "incx")]),
    
    # Level 2 BLAS
    ("sgemv", "void", [("char", "trans"), ("int", "m"), ("int", "n"),
                       ("float", "alpha"), ("const float*", "a"), ("int", "lda"),
                       ("const float*", "x"), ("int", "incx"),
                       ("float", "beta"), ("float*", "y"), ("int", "incy")]),
    ("dgemv", "void", [("char", "trans"), ("int", "m"), ("int", "n"),
                       ("double", "alpha"), ("const double*", "a"), ("int", "lda"),
                       ("const double*", "x"), ("int", "incx"),
                       ("double", "beta"), ("double*", "y"), ("int", "incy")]),
    
    # Level 3 BLAS
    ("sgemm", "void", [("char", "transa"), ("char", "transb"),
                       ("int", "m"), ("int", "n"), ("int", "k"),
                       ("float", "alpha"), ("const float*", "a"), ("int", "lda"),
                       ("const float*", "b"), ("int", "ldb"),
                       ("float", "beta"), ("float*", "c"), ("int", "ldc")]),
    ("dgemm", "void", [("char", "transa"), ("char", "transb"),
                       ("int", "m"), ("int", "n"), ("int", "k"),
                       ("double", "alpha"), ("const double*", "a"), ("int", "lda"),
                       ("const double*", "b"), ("int", "ldb"),
                       ("double", "beta"), ("double*", "c"), ("int", "ldc")]),
]


def generate_typedef(funcname: str, return_type: str, params: List[Tuple[str, str]]) -> str:
    """Generate a typedef for the Fortran function signature."""
    param_str = ", ".join(f"{ptype}* {pname}" for ptype, pname in params)
    return f"typedef {return_type} (*fblas_{funcname}_t)({param_str});"


def generate_cblas_wrapper(funcname: str, return_type: str, params: List[Tuple[str, str]]) -> str:
    """Generate a CBLAS-style wrapper function."""
    
    # Build function signature
    cblas_params = ", ".join(f"{ptype} {pname}" for ptype, pname in params)
    
    wrapper = f"""
/* CBLAS wrapper for {funcname} */
static {return_type} fb_blr_{funcname}({cblas_params}) {{
    static fblas_{funcname}_t {funcname}_fn = NULL;
    if (!{funcname}_fn) {{
        {funcname}_fn = (fblas_{funcname}_t)FB_GET_PROC_ADDRESS(g_blr_handle, "{funcname}_");
        if (!{funcname}_fn) {{
"""
    
    # Add error return based on return type
    if return_type == "void":
        wrapper += "            return;\n"
    elif "float" in return_type:
        wrapper += "            return 0.0f;\n"
    elif "double" in return_type:
        wrapper += "            return 0.0;\n"
    else:
        wrapper += f"            return ({return_type})0;\n"
    
    wrapper += "        }\n    }\n    "
    
    # Build call with parameter conversions
    call_params = []
    for ptype, pname in params:
        if ptype == "const float*" or ptype == "const double*" or ptype == "float*" or ptype == "double*":
            # Pointers pass through directly
            call_params.append(pname)
        elif ptype == "const float*" or ptype == "float*":
            # Scalar float - convert to pointer
            call_params.append(f"&{pname}_copy")
        elif ptype == "const double*" or ptype == "double*":
            # Scalar double - convert to pointer
            call_params.append(f"&{pname}_copy")
        elif ptype in ["int", "char"]:
            # Scalars passed by reference
            call_params.append(f"&{pname}_copy")
        else:
            call_params.append(f"&{pname}")
    
    call_str = ", ".join(call_params)
    
    # Add scalar copies before the call
    scalar_copies = []
    for ptype, pname in params:
        if ptype in ["int", "char", "float", "double"]:
            scalar_copies.append(f"{ptype} {pname}_copy = {pname};")
    
    if scalar_copies:
        wrapper += "\n    ".join(scalar_copies) + "\n    "
    
    # Make the actual call
    if return_type == "void":
        wrapper += f"{funcname}_fn({call_str});\n}}\n"
    else:
        wrapper += f"return {funcname}_fn({call_str});\n}}\n"
    
    return wrapper


def generate_wrapper_batch(operations: List[Tuple[str, str, List[Tuple[str, str]]]]) -> str:
    """Generate all typedefs and wrappers."""
    output = []
    
    # Generate all typedefs first
    output.append("/* ============================================================================\n")
    output.append(" * Fortran Function Typedefs (generated)\n")
    output.append(" * ========================================================================= */\n\n")
    
    for funcname, return_type, params in operations:
        output.append(generate_typedef(funcname, return_type, params))
        output.append("\n")
    
    # Then generate all wrappers
    output.append("\n/* ============================================================================\n")
    output.append(" * CBLAS Wrapper Functions (generated)\n")
    output.append(" * ========================================================================= */\n")
    
    for funcname, return_type, params in operations:
        output.append(generate_cblas_wrapper(funcname, return_type, params))
        output.append("\n")
    
    return "".join(output)


def main():
    """Main script entry point."""
    
    print("=" * 80)
    print("BLAS/LAPACK Reference Backend Wrapper Generator")
    print("=" * 80)
    print()
    
    # Generate the wrapper code
    print(f"Generating wrappers for {len(OPERATIONS)} operations...")
    wrapper_code = generate_wrapper_batch(OPERATIONS)
    
    # Print sample
    lines = wrapper_code.split('\n')
    print(f"Generated {len(lines)} lines of code")
    print()
    print("Sample output (first 30 lines):")
    print("-" * 80)
    for line in lines[:30]:
        print(line)
    print("-" * 80)
    print()
    
    # In real usage, this would be appended to blas_lapack_reference_backend.c
    print("TODO: Append this generated code to blas_lapack_reference_backend.c")
    print()
    print("NOTE: This is a proof-of-concept showing:")
    print("  1. How to generate 1248 wrapper functions automatically")
    print("  2. Pattern for converting CBLAS params to Fortran by-reference calls")
    print("  3. Proper typedef generation for each operation")
    print()
    print("For full integration, would need:")
    print("  - Complete operation list (from operations.h or database)")
    print("  - Batch append to C file with proper formatting")
    print("  - Symbol visibility attributes (static)")
    print("  - Documentation for each wrapper")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
