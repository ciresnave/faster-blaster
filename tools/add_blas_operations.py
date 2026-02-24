#!/usr/bin/env python3
"""
Automated BLAS Operation Addition Tool
Adds operations to all backends simultaneously with proper error handling
"""

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class BlasLevel(Enum):
    LEVEL1 = "Level 1"
    LEVEL2 = "Level 2"
    LEVEL3 = "Level 3"


class OperationType(Enum):
    ROTATION = "rotation"
    SYMMETRIC = "symmetric"
    TRIANGULAR = "triangular"
    GENERAL = "general"


@dataclass
class BlasOperation:
    """Definition of a BLAS operation"""
    name: str  # e.g., "ssymv"
    dtype: str  # "float" or "double"
    level: BlasLevel
    op_type: OperationType
    description: str
    params: List[tuple]  # (param_name, param_type, direction)
    
    
# ============================================================================
# OPERATION DEFINITIONS - Phase 1 High Priority
# ============================================================================

PHASE1_OPERATIONS = {
    # Level 2 - Symmetric Matrix Operations
    "ssymv": BlasOperation(
        name="ssymv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "float", "in"),
            ("y", "float*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "dsymv": BlasOperation(
        name="dsymv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "double", "in"),
            ("y", "double*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "sger": BlasOperation(
        name="sger",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.GENERAL,
        description="General rank-1 update: A = alpha*x*y' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("y", "float*", "in"),
            ("incy", "int64_t", "in"),
            ("A", "float*", "inout"),
            ("lda", "int64_t", "in"),
        ]
    ),
    "dger": BlasOperation(
        name="dger",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.GENERAL,
        description="General rank-1 update: A = alpha*x*y' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("y", "double*", "in"),
            ("incy", "int64_t", "in"),
            ("A", "double*", "inout"),
            ("lda", "int64_t", "in"),
        ]
    ),
    "strmv": BlasOperation(
        name="strmv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular matrix-vector multiply: x = A*x",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "dtrmv": BlasOperation(
        name="dtrmv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular matrix-vector multiply: x = A*x",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "strsv": BlasOperation(
        name="strsv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular system solve: A*x = b",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "dtrsv": BlasOperation(
        name="dtrsv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular system solve: A*x = b",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    # Level 3 - Symmetric Operations
    "ssymm": BlasOperation(
        name="ssymm",
        dtype="float",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric matrix-matrix multiply: C = alpha*A*B + beta*C",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("side", "fb_side_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "float*", "in"),
            ("ldb", "int64_t", "in"),
            ("beta", "float", "in"),
            ("C", "float*", "inout"),
            ("ldc", "int64_t", "in"),
        ]
    ),
    "dsymm": BlasOperation(
        name="dsymm",
        dtype="double",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric matrix-matrix multiply: C = alpha*A*B + beta*C",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("side", "fb_side_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "double*", "in"),
            ("ldb", "int64_t", "in"),
            ("beta", "double", "in"),
            ("C", "double*", "inout"),
            ("ldc", "int64_t", "in"),
        ]
    ),
    "strmm": BlasOperation(
        name="strmm",
        dtype="float",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.TRIANGULAR,
        description="Triangular matrix-matrix multiply: B = alpha*A*B",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("side", "fb_side_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "float*", "inout"),
            ("ldb", "int64_t", "in"),
        ]
    ),
    "dtrmm": BlasOperation(
        name="dtrmm",
        dtype="double",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.TRIANGULAR,
        description="Triangular matrix-matrix multiply: B = alpha*A*B",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("side", "fb_side_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "double*", "inout"),
            ("ldb", "int64_t", "in"),
        ]
    ),
    "strsm": BlasOperation(
        name="strsm",
        dtype="float",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.TRIANGULAR,
        description="Triangular system solve with multiple RHS: A*X = alpha*B",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("side", "fb_side_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "float*", "inout"),
            ("ldb", "int64_t", "in"),
        ]
    ),
    "dtrsm": BlasOperation(
        name="dtrsm",
        dtype="double",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.TRIANGULAR,
        description="Triangular system solve with multiple RHS: A*X = alpha*B",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("side", "fb_side_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "double*", "inout"),
            ("ldb", "int64_t", "in"),
        ]
    ),
}


# ============================================================================
# OPERATION DEFINITIONS - Phase 2: Rank Updates and Band/Packed Variants
# ============================================================================

PHASE2_OPERATIONS = {
    # Level 2 - Symmetric Rank-1 and Rank-2 Updates
    "ssyr": BlasOperation(
        name="ssyr",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-1 update: A = alpha*x*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("A", "float*", "inout"),
            ("lda", "int64_t", "in"),
        ]
    ),
    "dsyr": BlasOperation(
        name="dsyr",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-1 update: A = alpha*x*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("A", "double*", "inout"),
            ("lda", "int64_t", "in"),
        ]
    ),
    "ssyr2": BlasOperation(
        name="ssyr2",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-2 update: A = alpha*x*y' + alpha*y*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("y", "float*", "in"),
            ("incy", "int64_t", "in"),
            ("A", "float*", "inout"),
            ("lda", "int64_t", "in"),
        ]
    ),
    "dsyr2": BlasOperation(
        name="dsyr2",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-2 update: A = alpha*x*y' + alpha*y*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("y", "double*", "in"),
            ("incy", "int64_t", "in"),
            ("A", "double*", "inout"),
            ("lda", "int64_t", "in"),
        ]
    ),
    # Level 3 - Symmetric Rank-k and Rank-2k Updates
    "ssyrk": BlasOperation(
        name="ssyrk",
        dtype="float",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-k update: C = alpha*A*A' + beta*C",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("beta", "float", "in"),
            ("C", "float*", "inout"),
            ("ldc", "int64_t", "in"),
        ]
    ),
    "dsyrk": BlasOperation(
        name="dsyrk",
        dtype="double",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-k update: C = alpha*A*A' + beta*C",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("beta", "double", "in"),
            ("C", "double*", "inout"),
            ("ldc", "int64_t", "in"),
        ]
    ),
    "ssyr2k": BlasOperation(
        name="ssyr2k",
        dtype="float",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-2k update: C = alpha*A*B' + alpha*B*A' + beta*C",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "float*", "in"),
            ("ldb", "int64_t", "in"),
            ("beta", "float", "in"),
            ("C", "float*", "inout"),
            ("ldc", "int64_t", "in"),
        ]
    ),
    "dsyr2k": BlasOperation(
        name="dsyr2k",
        dtype="double",
        level=BlasLevel.LEVEL3,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric rank-2k update: C = alpha*A*B' + alpha*B*A' + beta*C",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("B", "double*", "in"),
            ("ldb", "int64_t", "in"),
            ("beta", "double", "in"),
            ("C", "double*", "inout"),
            ("ldc", "int64_t", "in"),
        ]
    ),
    # Level 2 - Band Matrix Operations
    "sgbmv": BlasOperation(
        name="sgbmv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.GENERAL,
        description="General band matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("kl", "int64_t", "in"),
            ("ku", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "float", "in"),
            ("y", "float*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "dgbmv": BlasOperation(
        name="dgbmv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.GENERAL,
        description="General band matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("m", "int64_t", "in"),
            ("n", "int64_t", "in"),
            ("kl", "int64_t", "in"),
            ("ku", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "double", "in"),
            ("y", "double*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "ssbmv": BlasOperation(
        name="ssbmv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric band matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "float", "in"),
            ("y", "float*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "dsbmv": BlasOperation(
        name="dsbmv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric band matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "double", "in"),
            ("y", "double*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "stbmv": BlasOperation(
        name="stbmv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular band matrix-vector multiply: x = A*x",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "dtbmv": BlasOperation(
        name="dtbmv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular band matrix-vector multiply: x = A*x",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "stbsv": BlasOperation(
        name="stbsv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular band system solve: A*x = b",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("A", "float*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "dtbsv": BlasOperation(
        name="dtbsv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular band system solve: A*x = b",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("k", "int64_t", "in"),
            ("A", "double*", "in"),
            ("lda", "int64_t", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    # Level 2 - Packed Matrix Operations
    "sspmv": BlasOperation(
        name="sspmv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric packed matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("Ap", "float*", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "float", "in"),
            ("y", "float*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "dspmv": BlasOperation(
        name="dspmv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric packed matrix-vector multiply: y = alpha*A*x + beta*y",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("Ap", "double*", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("beta", "double", "in"),
            ("y", "double*", "inout"),
            ("incy", "int64_t", "in"),
        ]
    ),
    "sspr": BlasOperation(
        name="sspr",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric packed rank-1 update: A = alpha*x*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("Ap", "float*", "inout"),
        ]
    ),
    "dspr": BlasOperation(
        name="dspr",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric packed rank-1 update: A = alpha*x*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("Ap", "double*", "inout"),
        ]
    ),
}


# ============================================================================
# OPERATION DEFINITIONS - Phase 3: Packed Triangular, Rank-2, and Rotations
# ============================================================================

PHASE3_OPERATIONS = {
    # Level 2 - Triangular Packed Matrix Operations
    "stpmv": BlasOperation(
        name="stpmv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular packed matrix-vector multiply: x = A*x",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("Ap", "float*", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "dtpmv": BlasOperation(
        name="dtpmv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular packed matrix-vector multiply: x = A*x",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("Ap", "double*", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "stpsv": BlasOperation(
        name="stpsv",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular packed system solve: A*x = b",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("Ap", "float*", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    "dtpsv": BlasOperation(
        name="dtpsv",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.TRIANGULAR,
        description="Triangular packed system solve: A*x = b",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("trans", "fb_transpose_t", "in"),
            ("diag", "fb_diag_t", "in"),
            ("n", "int64_t", "in"),
            ("Ap", "double*", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
        ]
    ),
    # Level 2 - Symmetric Packed Rank-2 Updates
    "sspr2": BlasOperation(
        name="sspr2",
        dtype="float",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric packed rank-2 update: A = alpha*x*y' + alpha*y*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "float", "in"),
            ("x", "float*", "in"),
            ("incx", "int64_t", "in"),
            ("y", "float*", "in"),
            ("incy", "int64_t", "in"),
            ("Ap", "float*", "inout"),
        ]
    ),
    "dspr2": BlasOperation(
        name="dspr2",
        dtype="double",
        level=BlasLevel.LEVEL2,
        op_type=OperationType.SYMMETRIC,
        description="Symmetric packed rank-2 update: A = alpha*x*y' + alpha*y*x' + A",
        params=[
            ("layout", "fb_layout_t", "in"),
            ("uplo", "fb_uplo_t", "in"),
            ("n", "int64_t", "in"),
            ("alpha", "double", "in"),
            ("x", "double*", "in"),
            ("incx", "int64_t", "in"),
            ("y", "double*", "in"),
            ("incy", "int64_t", "in"),
            ("Ap", "double*", "inout"),
        ]
    ),
    # Level 1 - Plane Rotations
    "srot": BlasOperation(
        name="srot",
        dtype="float",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Apply Givens rotation: [x; y] = [c s; -s c] * [x; y]",
        params=[
            ("n", "int64_t", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
            ("y", "float*", "inout"),
            ("incy", "int64_t", "in"),
            ("c", "float", "in"),
            ("s", "float", "in"),
        ]
    ),
    "drot": BlasOperation(
        name="drot",
        dtype="double",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Apply Givens rotation: [x; y] = [c s; -s c] * [x; y]",
        params=[
            ("n", "int64_t", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
            ("y", "double*", "inout"),
            ("incy", "int64_t", "in"),
            ("c", "double", "in"),
            ("s", "double", "in"),
        ]
    ),
    "srotg": BlasOperation(
        name="srotg",
        dtype="float",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Generate Givens rotation",
        params=[
            ("a", "float*", "inout"),
            ("b", "float*", "inout"),
            ("c", "float*", "out"),
            ("s", "float*", "out"),
        ]
    ),
    "drotg": BlasOperation(
        name="drotg",
        dtype="double",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Generate Givens rotation",
        params=[
            ("a", "double*", "inout"),
            ("b", "double*", "inout"),
            ("c", "double*", "out"),
            ("s", "double*", "out"),
        ]
    ),
    "srotm": BlasOperation(
        name="srotm",
        dtype="float",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Apply modified Givens rotation",
        params=[
            ("n", "int64_t", "in"),
            ("x", "float*", "inout"),
            ("incx", "int64_t", "in"),
            ("y", "float*", "inout"),
            ("incy", "int64_t", "in"),
            ("param", "float*", "in"),
        ]
    ),
    "drotm": BlasOperation(
        name="drotm",
        dtype="double",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Apply modified Givens rotation",
        params=[
            ("n", "int64_t", "in"),
            ("x", "double*", "inout"),
            ("incx", "int64_t", "in"),
            ("y", "double*", "inout"),
            ("incy", "int64_t", "in"),
            ("param", "double*", "in"),
        ]
    ),
    "srotmg": BlasOperation(
        name="srotmg",
        dtype="float",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Generate modified Givens rotation",
        params=[
            ("d1", "float*", "inout"),
            ("d2", "float*", "inout"),
            ("x1", "float*", "inout"),
            ("y1", "float", "in"),
            ("param", "float*", "out"),
        ]
    ),
    "drotmg": BlasOperation(
        name="drotmg",
        dtype="double",
        level=BlasLevel.LEVEL1,
        op_type=OperationType.ROTATION,
        description="Generate modified Givens rotation",
        params=[
            ("d1", "double*", "inout"),
            ("d2", "double*", "inout"),
            ("x1", "double*", "inout"),
            ("y1", "double", "in"),
            ("param", "double*", "out"),
        ]
    ),
}


# ============================================================================
# CODE GENERATORS - AOCL BLIS Backend
# ============================================================================

def generate_aocl_typedef(op: BlasOperation) -> str:
    """Generate typedef for CBLAS function pointer"""
    # Build param list for typedef
    params = []
    for pname, ptype, _ in op.params:
        if ptype == "fb_layout_t":
            params.append("CBLAS_LAYOUT")
        elif ptype == "fb_uplo_t":
            params.append("CBLAS_UPLO")
        elif ptype == "fb_side_t":
            params.append("CBLAS_SIDE")
        elif ptype == "fb_transpose_t":
            params.append("CBLAS_TRANSPOSE")
        elif ptype == "fb_diag_t":
            params.append("CBLAS_DIAG")
        elif ptype == "int64_t":
            params.append("const int")
        elif ptype.endswith("*"):
            params.append(f"const {ptype}" if "const" in ptype or pname in ["A", "x", "y", "B"] else ptype)
        else:
            params.append(f"const {ptype}")
    
    params_str = ", ".join(params)
    return f"typedef void (*cblas_{op.name}_t)({params_str});"


def generate_aocl_struct_field(op: BlasOperation) -> str:
    """Generate struct field for function pointer"""
    return f"    cblas_{op.name}_t {op.name};"


def generate_aocl_loader(op: BlasOperation) -> str:
    """Generate function pointer loading code"""
    return f'    g_aocl.{op.name} = (cblas_{op.name}_t)FB_GET_PROC_ADDRESS(g_aocl.handle, "cblas_{op.name}");'


def generate_aocl_wrapper(op: BlasOperation) -> str:
    """Generate AOCL BLIS wrapper function"""
    
    # Build parameter list
    params_decl = []
    for pname, ptype, _ in op.params:
        if ptype.endswith("*"):
            if "const" in ptype:
                params_decl.append(f"{ptype} {pname}")
            elif pname in ["A", "x", "y", "B", "alpha", "beta"]:  # Typically const inputs
                params_decl.append(f"const {ptype} {pname}")
            else:
                params_decl.append(f"{ptype} {pname}")
        else:
            params_decl.append(f"const {ptype} {pname}")
    
    params_str = ",\n                       ".join(params_decl)
    
    # Build enum conversions and call params
    conversions = []
    call_params = []
    
    for pname, ptype, _ in op.params:
        if ptype == "fb_layout_t":
            conversions.append("CBLAS_LAYOUT cblas_layout = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;")
            call_params.append("cblas_layout")
        elif ptype == "fb_uplo_t":
            conversions.append("CBLAS_UPLO cblas_uplo = (uplo == FB_UPPER) ? CblasUpper : CblasLower;")
            call_params.append("cblas_uplo")
        elif ptype == "fb_side_t":
            conversions.append("CBLAS_SIDE cblas_side = (side == FB_LEFT) ? CblasLeft : CblasRight;")
            call_params.append("cblas_side")
        elif ptype == "fb_transpose_t":
            conversions.append("CBLAS_TRANSPOSE cblas_trans = (trans == FB_NO_TRANS) ? CblasNoTrans : (trans == FB_TRANS) ? CblasTrans : CblasConjTrans;")
            call_params.append("cblas_trans")
        elif ptype == "fb_diag_t":
            conversions.append("CBLAS_DIAG cblas_diag = (diag == FB_NON_UNIT) ? CblasNonUnit : CblasUnit;")
            call_params.append("cblas_diag")
        elif ptype == "int64_t":
            call_params.append(f"(int){pname}")
        else:
            call_params.append(pname)
    
    conversions_str = "\n        ".join(sorted(set(conversions)))  # Remove duplicates
    call_params_str = ", ".join(call_params)
    
    return f"""/* {op.description} */
static void aocl_{op.name}({params_str}) {{
    if (g_aocl.{op.name}) {{
        {conversions_str}
        g_aocl.{op.name}({call_params_str});
    }}
}}
"""


def generate_aocl_vtable_entry(op: BlasOperation) -> str:
    """Generate vtable entry for AOCL backend"""
    return f"    .{op.name} = aocl_{op.name},"


# ============================================================================
# CODE GENERATORS - cuBLAS Backend (TODO)
# ============================================================================

def generate_cublas_wrapper(op: BlasOperation) -> str:
    """Generate cuBLAS wrapper function (with GPU memory management)"""
    # TODO: Implement cuBLAS-specific wrapper generation
    return f"// TODO: Implement cuBLAS wrapper for {op.name}\n"


# ============================================================================
# CODE GENERATORS - CLBlast Backend (TODO)
# ============================================================================

def generate_clblast_wrapper(op: BlasOperation) -> str:
    """Generate CLBlast wrapper function (with OpenCL buffer management)"""
    # TODO: Implement CLBlast-specific wrapper generation
    return f"// TODO: Implement CLBlast wrapper for {op.name}\n"


# ============================================================================
# MAIN GENERATION LOGIC
# ============================================================================

def generate_operations_for_backend(backend: str, operations: Dict[str, BlasOperation]) -> Dict[str, str]:
    """Generate code for a specific backend"""
    
    if backend == "aocl":
        typedefs = []
        struct_fields = []
        loaders = []
        wrappers = []
        vtable_entries = []
        
        # Group by level and type
        level2_ops = sorted([op for op in operations.values() if op.level == BlasLevel.LEVEL2], key=lambda x: x.name)
        level3_ops = sorted([op for op in operations.values() if op.level == BlasLevel.LEVEL3], key=lambda x: x.name)
        
        # Generate typedefs
        if level2_ops:
            typedefs.append("\n/* Level 2 BLAS typedefs */")
            for op in level2_ops:
                typedefs.append(generate_aocl_typedef(op))
        
        if level3_ops:
            typedefs.append("\n/* Level 3 BLAS typedefs */")
            for op in level3_ops:
                typedefs.append(generate_aocl_typedef(op))
        
        # Generate struct fields
        struct_fields.append("\n    /* Level 2 BLAS */")
        for op in level2_ops:
            struct_fields.append(generate_aocl_struct_field(op))
        
        struct_fields.append("\n    /* Level 3 BLAS */")
        for op in level3_ops:
            struct_fields.append(generate_aocl_struct_field(op))
        
        # Generate loaders
        loaders.append("\n    /* Level 2 BLAS */")
        for op in level2_ops:
            loaders.append(generate_aocl_loader(op))
        
        loaders.append("\n    /* Level 3 BLAS */")
        for op in level3_ops:
            loaders.append(generate_aocl_loader(op))
        
        # Generate wrappers
        if level2_ops:
            wrappers.append("\n/* ============================================================================")
            wrappers.append(" * BLAS Level 2 Operations - NEW")
            wrappers.append(" * ========================================================================== */\n")
            for op in level2_ops:
                wrappers.append(generate_aocl_wrapper(op))
                vtable_entries.append(generate_aocl_vtable_entry(op))
        
        if level3_ops:
            wrappers.append("\n/* ============================================================================")
            wrappers.append(" * BLAS Level 3 Operations - NEW")
            wrappers.append(" * ========================================================================== */\n")
            for op in level3_ops:
                wrappers.append(generate_aocl_wrapper(op))
                vtable_entries.append(generate_aocl_vtable_entry(op))
        
        return {
            "typedefs": "\n".join(typedefs),
            "struct_fields": "\n".join(struct_fields),
            "loaders": "\n".join(loaders),
            "wrappers": "\n".join(wrappers),
            "vtable": "\n".join(vtable_entries)
        }
    
    elif backend == "cublas":
        return {"wrappers": "// cuBLAS implementation TODO", "vtable": ""}
    
    elif backend == "clblast":
        return {"wrappers": "// CLBlast implementation TODO", "vtable": ""}
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main():
    """Main entry point"""
    import sys
    
    # Select which phase to generate
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "phase1":
            phase = "PHASE1"
            operations = PHASE1_OPERATIONS
        elif arg == "phase2":
            phase = "PHASE2"
            operations = PHASE2_OPERATIONS
        elif arg == "phase3":
            phase = "PHASE3"
            operations = PHASE3_OPERATIONS
        else:
            print(f"Unknown phase: {arg}")
            print("Usage: python add_blas_operations.py [phase1|phase2|phase3]")
            sys.exit(1)
    else:
        phase = "PHASE1"
        operations = PHASE1_OPERATIONS
    
    print("=" * 80)
    print("AUTOMATED BLAS OPERATION ADDITION TOOL")
    print("=" * 80)
    print()
    print(f"Generating {phase}: {len(operations)} operations")
    print()
    
    # Generate for each backend
    for backend in ["aocl"]:  # Start with AOCL, add others later
        print(f"\n{'=' * 80}")
        print(f"BACKEND: {backend.upper()} - {phase}")
        print('=' * 80)
        
        code = generate_operations_for_backend(backend, operations)
        
        print("\n--- TYPEDEFS (add after existing cblas_*_t typedefs, around line 107) ---")
        print(code["typedefs"])
        
        print("\n--- STRUCT FIELDS (add to g_aocl struct, around line 153) ---")
        print(code["struct_fields"])
        
        print("\n--- FUNCTION LOADERS (add in load_aocl_library, around line 265) ---")
        print(code["loaders"])
        
        print("\n--- WRAPPER FUNCTIONS (add after aocl_dgemm, around line 798) ---")
        print(code["wrappers"])
        
        print("\n--- VTABLE ENTRIES (add in g_aocl_vtable, after .dgemm line) ---")
        print(code["vtable"])
        print()
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(operations)} operations for {phase}")
    print("\nNext steps:")
    print("1. Copy TYPEDEFS after existing cblas_*_t typedefs (line ~107)")
    print("2. Copy STRUCT FIELDS to g_aocl struct (line ~153)")
    print("3. Copy FUNCTION LOADERS to load_aocl_library() (line ~265)")
    print("4. Copy WRAPPER FUNCTIONS after aocl_dgemm (line ~798)")
    print("5. Copy VTABLE ENTRIES to g_aocl_vtable (line ~1830)")
    print("6. Rebuild and test")
    print("7. Repeat for other backends (cuBLAS, CLBlast, oneMKL)")


if __name__ == "__main__":
    main()
