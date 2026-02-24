#!/usr/bin/env python3
"""
Code generator for BLAS operation wrappers
Generates implementation functions for AOCL BLIS backend
"""

import os
from typing import List, Tuple

# Level 1 BLAS operations to implement
LEVEL1_ROTATION_OPS = [
    ("srot", "float", "Apply plane rotation to two vectors"),
    ("drot", "double", "Apply plane rotation to two vectors"),
    ("srotg", "float", "Generate plane rotation parameters"),
    ("drotg", "double", "Generate plane rotation parameters"),
    ("srotm", "float", "Apply modified plane rotation"),
    ("drotm", "double", "Apply modified plane rotation"),
    ("srotmg", "float", "Generate modified plane rotation parameters"),
    ("drotmg", "double", "Generate modified plane rotation parameters"),
]

# Level 2 BLAS operations - Symmetric
LEVEL2_SYMMETRIC_OPS = [
    ("ssymv", "float", "Symmetric matrix-vector multiply"),
    ("dsymv", "double", "Symmetric matrix-vector multiply"),
    ("sger", "float", "General rank-1 update: A = alpha*x*y' + A"),
    ("dger", "double", "General rank-1 update: A = alpha*x*y' + A"),
    ("ssyr", "float", "Symmetric rank-1 update: A = alpha*x*x' + A"),
    ("dsyr", "double", "Symmetric rank-1 update: A = alpha*x*x' + A"),
    ("ssyr2", "float", "Symmetric rank-2 update"),
    ("dsyr2", "double", "Symmetric rank-2 update"),
]

# Level 2 BLAS operations - Triangular
LEVEL2_TRIANGULAR_OPS = [
    ("strmv", "float", "Triangular matrix-vector multiply"),
    ("dtrmv", "double", "Triangular matrix-vector multiply"),
    ("strsv", "float", "Triangular system solve"),
    ("dtrsv", "double", "Triangular system solve"),
]

# Level 3 BLAS operations - Symmetric
LEVEL3_SYMMETRIC_OPS = [
    ("ssymm", "float", "Symmetric matrix-matrix multiply"),
    ("dsymm", "double", "Symmetric matrix-matrix multiply"),
    ("ssyrk", "float", "Symmetric rank-k update"),
    ("dsyrk", "double", "Symmetric rank-k update"),
    ("ssyr2k", "float", "Symmetric rank-2k update"),
    ("dsyr2k", "double", "Symmetric rank-2k update"),
]

# Level 3 BLAS operations - Triangular
LEVEL3_TRIANGULAR_OPS = [
    ("strmm", "float", "Triangular matrix-matrix multiply"),
    ("dtrmm", "double", "Triangular matrix-matrix multiply"),
    ("strsm", "float", "Triangular system solve with multiple RHS"),
    ("dtrsm", "double", "Triangular system solve with multiple RHS"),
]


def generate_symv_wrapper(op: str, dtype: str) -> str:
    """Generate ssymv/dsymv wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_uplo_t uplo, const int64_t n,
                       const {dtype} alpha, const {dtype} *A, const int64_t lda,
                       const {dtype} *x, const int64_t incx, const {dtype} beta,
                       {dtype} *y, const int64_t incy) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    
    cblas_{op}(cblas_layout, cblas_uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}}
"""


def generate_ger_wrapper(op: str, dtype: str) -> str:
    """Generate sger/dger wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const int64_t m, const int64_t n,
                       const {dtype} alpha, const {dtype} *x, const int64_t incx,
                       const {dtype} *y, const int64_t incy, {dtype} *A, const int64_t lda) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    
    cblas_{op}(cblas_layout, m, n, alpha, x, incx, y, incy, A, lda);
}}
"""


def generate_syr_wrapper(op: str, dtype: str) -> str:
    """Generate ssyr/dsyr wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_uplo_t uplo, const int64_t n,
                       const {dtype} alpha, const {dtype} *x, const int64_t incx,
                       {dtype} *A, const int64_t lda) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    
    cblas_{op}(cblas_layout, cblas_uplo, n, alpha, x, incx, A, lda);
}}
"""


def generate_trmv_wrapper(op: str, dtype: str) -> str:
    """Generate strmv/dtrmv wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int64_t n,
                       const {dtype} *A, const int64_t lda, {dtype} *x, const int64_t incx) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans :
                                    (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    
    cblas_{op}(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, A, lda, x, incx);
}}
"""


def generate_trsv_wrapper(op: str, dtype: str) -> str:
    """Generate strsv/dtrsv wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int64_t n,
                       const {dtype} *A, const int64_t lda, {dtype} *x, const int64_t incx) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans :
                                    (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    
    cblas_{op}(cblas_layout, cblas_uplo, cblas_trans, cblas_diag, n, A, lda, x, incx);
}}
"""


def generate_symm_wrapper(op: str, dtype: str) -> str:
    """Generate ssymm/dsymm wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const int64_t m, const int64_t n, const {dtype} alpha,
                       const {dtype} *A, const int64_t lda, const {dtype} *B, const int64_t ldb,
                       const {dtype} beta, {dtype} *C, const int64_t ldc) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    
    cblas_{op}(cblas_layout, cblas_side, cblas_uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}}
"""


def generate_syrk_wrapper(op: str, dtype: str) -> str:
    """Generate ssyrk/dsyrk wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans,
                       const int64_t n, const int64_t k, const {dtype} alpha,
                       const {dtype} *A, const int64_t lda, const {dtype} beta,
                       {dtype} *C, const int64_t ldc) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans :
                                    (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
    
    cblas_{op}(cblas_layout, cblas_uplo, cblas_trans, n, k, alpha, A, lda, beta, C, ldc);
}}
"""


def generate_trmm_wrapper(op: str, dtype: str) -> str:
    """Generate strmm/dtrmm wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int64_t m,
                       const int64_t n, const {dtype} alpha, const {dtype} *A, const int64_t lda,
                       {dtype} *B, const int64_t ldb) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans :
                                    (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    
    cblas_{op}(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, A, lda, B, ldb);
}}
"""


def generate_trsm_wrapper(op: str, dtype: str) -> str:
    """Generate strsm/dtrsm wrapper implementation"""
    return f"""
static void aocl_{op}(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo,
                       const fb_transpose_t trans, const fb_diag_t diag, const int64_t m,
                       const int64_t n, const {dtype} alpha, const {dtype} *A, const int64_t lda,
                       {dtype} *B, const int64_t ldb) {{
    CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
    CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans :
                                    (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;
    CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;
    
    cblas_{op}(cblas_layout, cblas_side, cblas_uplo, cblas_trans, cblas_diag, m, n, alpha, A, lda, B, ldb);
}}
"""


def generate_all_wrappers():
    """Generate all wrapper implementations"""
    output = []
    
    output.append("/* ===================================================================")
    output.append(" * BLAS Level 2 - Symmetric Matrix Operations")
    output.append(" * =================================================================== */\n")
    
    for op, dtype, desc in LEVEL2_SYMMETRIC_OPS:
        if "symv" in op:
            output.append(f"/* {desc} */")
            output.append(generate_symv_wrapper(op, dtype))
        elif "ger" in op and "2" not in op:
            output.append(f"/* {desc} */")
            output.append(generate_ger_wrapper(op, dtype))
        elif "syr" in op and "2" not in op and "k" not in op:
            output.append(f"/* {desc} */")
            output.append(generate_syr_wrapper(op, dtype))
    
    output.append("\n/* ===================================================================")
    output.append(" * BLAS Level 2 - Triangular Matrix Operations")
    output.append(" * =================================================================== */\n")
    
    for op, dtype, desc in LEVEL2_TRIANGULAR_OPS:
        if "trmv" in op:
            output.append(f"/* {desc} */")
            output.append(generate_trmv_wrapper(op, dtype))
        elif "trsv" in op:
            output.append(f"/* {desc} */")
            output.append(generate_trsv_wrapper(op, dtype))
    
    output.append("\n/* ===================================================================")
    output.append(" * BLAS Level 3 - Symmetric Matrix Operations")
    output.append(" * =================================================================== */\n")
    
    for op, dtype, desc in LEVEL3_SYMMETRIC_OPS:
        if "symm" in op:
            output.append(f"/* {desc} */")
            output.append(generate_symm_wrapper(op, dtype))
        elif "syrk" in op and "2" not in op:
            output.append(f"/* {desc} */")
            output.append(generate_syrk_wrapper(op, dtype))
    
    output.append("\n/* ===================================================================")
    output.append(" * BLAS Level 3 - Triangular Matrix Operations")
    output.append(" * =================================================================== */\n")
    
    for op, dtype, desc in LEVEL3_TRIANGULAR_OPS:
        if "trmm" in op:
            output.append(f"/* {desc} */")
            output.append(generate_trmm_wrapper(op, dtype))
        elif "trsm" in op:
            output.append(f"/* {desc} */")
            output.append(generate_trsm_wrapper(op, dtype))
    
    return "\n".join(output)


def generate_vtable_entries():
    """Generate vtable assignment entries"""
    output = []
    
    output.append("    /* Level 2 - Symmetric operations */")
    for op, _, _ in LEVEL2_SYMMETRIC_OPS:
        output.append(f"    .{op} = aocl_{op},")
    
    output.append("\n    /* Level 2 - Triangular operations */")
    for op, _, _ in LEVEL2_TRIANGULAR_OPS:
        output.append(f"    .{op} = aocl_{op},")
    
    output.append("\n    /* Level 3 - Symmetric operations */")
    for op, _, _ in LEVEL3_SYMMETRIC_OPS:
        output.append(f"    .{op} = aocl_{op},")
    
    output.append("\n    /* Level 3 - Triangular operations */")
    for op, _, _ in LEVEL3_TRIANGULAR_OPS:
        output.append(f"    .{op} = aocl_{op},")
    
    return "\n".join(output)


if __name__ == "__main__":
    print("=" * 70)
    print("BLAS WRAPPER CODE GENERATOR")
    print("=" * 70)
    print()
    
    print("Generating wrapper implementations...")
    wrappers = generate_all_wrappers()
    
    print("\n" + "=" * 70)
    print("GENERATED WRAPPER FUNCTIONS")
    print("=" * 70)
    print(wrappers)
    
    print("\n\n" + "=" * 70)
    print("VTABLE ENTRIES TO ADD")
    print("=" * 70)
    print(generate_vtable_entries())
    
    print("\n\nGeneration complete!")
    print(f"Generated {len(LEVEL2_SYMMETRIC_OPS + LEVEL2_TRIANGULAR_OPS + LEVEL3_SYMMETRIC_OPS + LEVEL3_TRIANGULAR_OPS)} operations")
