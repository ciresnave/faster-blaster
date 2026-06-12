# faster-blaster Documentation

This directory contains all project documentation organized by category.

## 📖 Core Documentation

**Essential reading for understanding the project:**

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [HYBRID_DISPATCH_ARCHITECTURE.md](HYBRID_DISPATCH_ARCHITECTURE.md) - CPU/GPU hybrid dispatch design
- [BACKEND_OPERATIONS_SUPERSET.md](BACKEND_OPERATIONS_SUPERSET.md) - Complete catalog of 341 operations

## 🚀 Getting Started

**For users setting up and using faster-blaster:**

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide (5 minutes)
- [BUILD.md](BUILD.md) - Build instructions
- [INTEL_ONEMKL_SETUP.md](INTEL_ONEMKL_SETUP.md) - Intel oneAPI/oneMKL setup
- [ROCM_SETUP.md](ROCM_SETUP.md) - AMD ROCm setup

## 🔧 Developer Guides

**For developers extending or modifying faster-blaster:**

- [custom_backend_example.md](custom_backend_example.md) - How to add a new backend
- [backend_plugin_architecture.md](backend_plugin_architecture.md) - Plugin system design
- [OPERATION_CHAIN_COMPILER.md](OPERATION_CHAIN_COMPILER.md) - Operation fusion system

## 📊 Status & Implementation

**Current project status and completion tracking:**

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Interface completion status
- [../ROADMAP.md](../ROADMAP.md) - Development roadmap and timeline

## 🗃️ Archive

**Historical documents and temporary status files (kept for reference):**

The following files represent intermediate planning stages, temporary status snapshots, or superseded documentation. They are retained for historical context but are no longer actively maintained:

### Planning & Roadmap Documents (Superseded by ROADMAP.md)
- `IMPLEMENTATION_ROADMAP.md` - Early roadmap (superseded by root ROADMAP.md)
- `NEXT_STEPS_COMPLETE.md` - Temporary next steps snapshot
- `PRE_DEPLOYMENT_CHECKLIST.md` - Early deployment checklist
- `PROGRESS.md` - Temporary progress tracking

### Status Snapshots (Point-in-time, now outdated)
- `BACKEND_STATUS.md` - Backend status snapshot
- `BACKEND_EXPANSION_STATUS.md` - Backend expansion progress
- `BACKEND_IMPLEMENTATION_SUMMARY.md` - Implementation summary snapshot
- `COMPILATION_STATUS.md` - Compilation status snapshot
- `CPU_BACKENDS_STATUS.md` - CPU backend status
- `CUBLAS_IMPLEMENTATION_COMPLETE.md` - cuBLAS completion snapshot
- `GPU_BACKEND_STATUS.md` - GPU backend status
- `GPU_BACKEND_TEST_STATUS.md` - GPU test status
- `GPU_TRAIT_IMPLEMENTATION_SUMMARY.md` - GPU trait summary
- `MULTI_VENDOR_GPU_SUPPORT.md` - Multi-vendor GPU snapshot
- `MULTI_VENDOR_LAPACK_COMPLETE.md` - LAPACK completion snapshot
- `STATUS.md` - General status snapshot
- `HYBRID_DISPATCH_COMPLETE.md` - Hybrid dispatch completion summary

### Architecture Evolution (Early designs)
- `COMPLETE_ARCHITECTURE.md` - Early complete architecture
- `GPU_TRAIT_ARCHITECTURE.md` - GPU trait early design
- `GPU_BACKEND_STRATEGY.md` - GPU backend strategy
- `LAPACK_SOLVER_STRATEGY.md` - LAPACK solver strategy
- `UNIFIED_GPU_BACKEND.md` - Unified GPU backend design
- `ZERO_COST_LAPACK_ABSTRACTION.md` - Zero-cost abstraction design

### Implementation Details (Specific components)
- `BACKEND_GENERATION.md` - Backend code generation
- `BACKEND_OPERATIONS_INVENTORY.md` - Operations inventory
- `WINDOWS_ROCM_REALITY.md` - Windows ROCm notes

## 📋 Documentation Standards

When adding new documentation:

1. **Core docs** → Place in root `docs/` with clear, descriptive names
2. **Setup guides** → Prefix with platform/vendor name (e.g., `INTEL_ONEMKL_SETUP.md`)
3. **Status/temporary** → Mark clearly as temporary in title, will be archived
4. **Architecture** → Use `_ARCHITECTURE` suffix for design documents

## 🔍 Finding Documentation

**Looking for something specific?**

- **"How do I build this?"** → [BUILD.md](BUILD.md)
- **"How does it work?"** → [ARCHITECTURE.md](ARCHITECTURE.md)  
- **"What operations are supported?"** → [BACKEND_OPERATIONS_SUPERSET.md](BACKEND_OPERATIONS_SUPERSET.md)
- **"What's the plan?"** → [../ROADMAP.md](../ROADMAP.md)
- **"How do I add a backend?"** → [custom_backend_example.md](custom_backend_example.md)
- **"Setup Intel MKL/oneAPI"** → [INTEL_ONEMKL_SETUP.md](INTEL_ONEMKL_SETUP.md)
- **"Setup AMD ROCm"** → [ROCM_SETUP.md](ROCM_SETUP.md)
