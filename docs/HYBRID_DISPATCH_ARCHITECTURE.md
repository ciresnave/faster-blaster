# Hybrid Compile-Time + Runtime Dispatch Architecture

**Vision**: Treat CPU and GPU as "compute devices with attached memory", with compile-time optimization within each device and runtime selection between devices.

## Design Principles

1. **Compile-Time Optimization**: Zero-cost abstraction within each backend (cuBLAS, MKL, etc.)
2. **Runtime Device Selection**: Choose best available device based on current load/availability
3. **Data-Aware Scheduling**: Minimize data transfers, prefer device where data resides
4. **Operation-Aware Dispatch**: Small ops may be faster on CPU, large on GPU

## Architecture Layers

### Layer 1: Backend Traits (Compile-Time Optimized)

Each backend provides a trait with 341 operations:

```c
// GPU backend trait (already exists: gpu_backend_trait.h)
typedef struct fb_gpu_backend_trait {
    const char* name;
    fb_device_type_t type;  // FB_DEVICE_GPU
    
    // 341 operations:
    // - 152 standard BLAS (Level 1/2/3)
    // - 116 LAPACK (28 current + 88 extended)
    // - 44 batched/strided BLAS
    // - 29 fused operations
    
    void (*saxpy)(...);
    void (*sgemm)(...);
    int (*sgetrf)(...);
    int (*sgemm_batched)(...);
    // ... all 341 operations
} fb_gpu_backend_trait_t;

// CPU backend trait (TO CREATE: cpu_backend_trait.h)
typedef struct fb_cpu_backend_trait {
    const char* name;
    fb_device_type_t type;  // FB_DEVICE_CPU
    
    // Same 341 operations for interchangeability:
    void (*saxpy)(...);
    void (*sgemm)(...);
    int (*sgetrf)(...);
    
    // CPU-specific operations (optional):
    int (*sgbsv)(...);   // Band matrix solver
    int (*spbsv)(...);   // Packed Cholesky solver
    // ~270 additional CPU-specific ops for specialized storage formats
} fb_cpu_backend_trait_t;
```

### Layer 2: Compute Device Abstraction

Unified interface for any compute device:

```c
typedef enum {
    FB_DEVICE_CPU,
    FB_DEVICE_GPU,
    FB_DEVICE_FPGA,    // Future
    FB_DEVICE_NPU      // Future: Neural Processing Unit
} fb_device_type_t;

typedef struct {
    // Device identity
    int device_id;              // Physical device ID
    fb_device_type_t type;      // CPU or GPU
    char name[128];             // "NVIDIA RTX 4090", "AMD Ryzen 9 7950X", etc.
    
    // Backend trait (compile-time optimized)
    const void* trait;          // Points to fb_gpu_backend_trait_t or fb_cpu_backend_trait_t
    void* backend_handle;       // cuBLAS handle, MKL handle, etc.
    
    // Memory hierarchy
    size_t total_memory;        // Total device memory
    size_t available_memory;    // Currently available
    bool unified_memory;        // True for modern AMD APUs, Apple M-series
    
    // Performance characteristics
    float peak_gflops_fp32;     // Peak GFLOPS (single precision)
    float peak_gflops_fp64;     // Peak GFLOPS (double precision)
    float memory_bandwidth_gbs; // GB/s
    float pcie_bandwidth_gbs;   // PCIe bandwidth (0 for CPU)
    
    // Runtime state
    float current_load;         // 0.0 = idle, 1.0 = saturated
    int pending_operations;     // Queued operations
    bool power_constrained;     // Thermal throttling or battery mode
    
    // Operation cost model (from calibration)
    fb_operation_cost_table_t* cost_model;
    
} fb_compute_device_t;
```

### Layer 3: Smart Compute Manager

Manages all compute devices and schedules work:

```c
typedef enum {
    FB_POLICY_FASTEST,          // Always use fastest device (ignore load)
    FB_POLICY_LOAD_BALANCED,    // Distribute work based on current load
    FB_POLICY_DATA_LOCALITY,    // Prefer device where data already resides
    FB_POLICY_POWER_EFFICIENT,  // Prefer CPU on battery, GPU on AC power
    FB_POLICY_COST_AWARE        // Consider data transfer costs
} fb_scheduler_policy_t;

typedef struct {
    // All available compute devices
    fb_compute_device_t* devices;
    int num_devices;
    
    // Scheduling policy
    fb_scheduler_policy_t policy;
    
    // Data location tracking (optional optimization)
    fb_data_tracker_t* data_tracker;  // Tracks which device has which data
    
} fb_compute_manager_t;

// Initialize and detect all devices
fb_compute_manager_t* fb_compute_manager_init(void);

// Get device by type and index
fb_compute_device_t* fb_get_device(fb_compute_manager_t* mgr, 
                                    fb_device_type_t type, 
                                    int device_index);

// Smart device selection for specific operation
fb_compute_device_t* fb_select_device(
    fb_compute_manager_t* mgr,
    fb_operation_id_t operation,
    size_t m, size_t n, size_t k,  // Operation dimensions
    fb_data_location_t data_location,
    float* estimated_time_ms);      // Output: estimated execution time
```

### Layer 4: Smart Dispatch API

Two usage modes:

#### Mode 1: Explicit Device Selection (User Control)
```c
// User explicitly chooses device
fb_compute_device_t* my_gpu = fb_get_device(mgr, FB_DEVICE_GPU, 0);
fb_compute_device_t* my_cpu = fb_get_device(mgr, FB_DEVICE_CPU, 0);

// Use GPU
fb_gpu_backend_trait_t* gpu_trait = (fb_gpu_backend_trait_t*)my_gpu->trait;
gpu_trait->sgemm(...);

// Use CPU
fb_cpu_backend_trait_t* cpu_trait = (fb_cpu_backend_trait_t*)my_cpu->trait;
cpu_trait->sgemm(...);
```

#### Mode 2: Smart Automatic Dispatch (System Chooses)
```c
// System chooses best device for THIS invocation
void fb_sgemm_auto(fb_compute_manager_t* mgr,
                   char transa, char transb,
                   int m, int n, int k,
                   float alpha, const float* A, int lda,
                   const float* B, int ldb,
                   float beta, float* C, int ldc);

// Implementation:
void fb_sgemm_auto(...) {
    // 1. Estimate cost on each device
    //    - Consider operation size (m*n*k flops)
    //    - Consider current device load
    //    - Consider data transfer costs
    
    // 2. Select best device
    fb_compute_device_t* best = fb_select_device(
        mgr, FB_OP_SGEMM, m, n, k, 
        fb_get_data_location(A, B, C), NULL);
    
    // 3. Dispatch to compile-time optimized backend
    if (best->type == FB_DEVICE_GPU) {
        fb_gpu_backend_trait_t* trait = (fb_gpu_backend_trait_t*)best->trait;
        
        // Handle data transfers if needed
        fb_gpu_ptr_t d_A, d_B, d_C;
        trait->malloc(&d_A, m*k*sizeof(float));
        trait->memcpy_h2d(d_A, A, m*k*sizeof(float));
        // ... etc
        
        trait->sgemm(best->backend_handle, NULL, transa, transb, 
                     m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    } else {
        fb_cpu_backend_trait_t* trait = (fb_cpu_backend_trait_t*)best->trait;
        trait->sgemm(best->backend_handle, transa, transb,
                     m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}
```

## Example: Intelligent Load Balancing

```c
// Setup
fb_compute_manager_t* mgr = fb_compute_manager_init();
fb_set_scheduler_policy(mgr, FB_POLICY_LOAD_BALANCED);

// Scenario: NVIDIA GPU (fastest) is busy, AMD GPU is idle
// System automatically uses AMD GPU instead of waiting

float A[1024*1024], B[1024*1024], C[1024*1024];

// First call: cuBLAS on NVIDIA (fastest when idle)
fb_sgemm_auto(mgr, 'N', 'N', 1024, 1024, 1024, 
              1.0f, A, 1024, B, 1024, 0.0f, C, 1024);
// -> Uses NVIDIA RTX 4090 (cuBLAS) - device load: 0.0 -> 0.7

// Second call (NVIDIA still busy): rocBLAS on AMD
fb_sgemm_auto(mgr, 'N', 'N', 1024, 1024, 1024,
              1.0f, A, 1024, B, 1024, 0.0f, C, 1024);
// -> NVIDIA load = 0.7, AMD load = 0.0
// -> System chooses AMD RX 7900 XTX (rocBLAS) instead of waiting
// -> AMD gets work immediately, better system utilization!

// Third call (small matrix): CPU is actually faster!
fb_sgemm_auto(mgr, 'N', 'N', 32, 32, 32,
              1.0f, A, 32, B, 32, 0.0f, C, 32);
// -> Small matrix: GPU kernel launch overhead > compute time
// -> System uses CPU (OpenBLAS) - no PCIe transfer, lower latency
```

## Unified Operation Coverage

### Core Operations (Same on CPU and GPU) - 341 operations

All devices should support:
- **152 BLAS Level 1/2/3**: saxpy, sgemm, strsm, etc.
- **116 LAPACK**: sgetrf, spotrf, sgeev, sgesvd, etc.
- **44 Batched Operations**: sgemm_batched, strided_batched, etc.
- **29 Fused Operations**: gemm3m, gemmt, geam, syrkx, etc.

**Implementation Details**:
- GPU: Already defined in `gpu_backend_trait.h` (needs completion)
- CPU: Need to create `cpu_backend_trait.h` with same operations
- Batched ops on CPU: Sequential execution (backends may have optimized paths)
- Fused ops on CPU: May fall back to separate calls (MKL has optimized versions)

### CPU-Specific Extensions (Optional) - ~270 operations

Additional CPU-only operations for specialized storage:
- **Band matrices**: sgbsv, dgbtrf (tridiagonal, banded LU)
- **Packed storage**: spp, ssp operations (save 50% memory)
- **Symmetric band**: spbsv, spbtrf
- **Tridiagonal**: sgtsv, sgtts2

**Why optional**: These storage formats are less common on GPUs (irregular memory access patterns).

## Data Transfer Considerations

```c
typedef struct {
    void* ptr;                     // Memory pointer
    fb_compute_device_t* location; // Which device has this data
    size_t size;                   // Data size in bytes
    bool pinned;                   // Pinned/page-locked for faster transfers
} fb_tracked_allocation_t;

// Track allocations to avoid unnecessary transfers
fb_data_tracker_t* tracker = fb_data_tracker_create();
fb_track_allocation(tracker, A, sizeof(A), cpu_device);

// When dispatching, system knows data location
fb_compute_device_t* best = fb_select_device_with_data_costs(
    mgr, FB_OP_SGEMM, m, n, k, 
    fb_query_location(tracker, A),  // A is on CPU
    fb_query_location(tracker, B),  // B is on CPU
    fb_query_location(tracker, C)); // C is on CPU

// If selecting GPU: factor in PCIe transfer time
// Estimated time = transfer_time + compute_time
// Compare with CPU direct execution time
```

## Benefits of Hybrid Approach

### vs Pure Compile-Time (Current faster-blaster)
✅ **Better resource utilization**: Use idle devices instead of waiting
✅ **Multi-device workflows**: Can use CPU + multiple GPUs simultaneously  
✅ **Adaptive to system state**: Respond to thermal throttling, battery mode
✅ **Data locality awareness**: Avoid expensive transfers when possible

### vs Pure Runtime (FlexiBLAS)
✅ **Zero overhead within device**: Compile-time optimization per backend
✅ **Better performance**: Direct function calls, no vtable lookup overhead
✅ **Smaller binary**: Can exclude unused backends at compile time
✅ **Cross-device selection**: FlexiBLAS only switches between CPU libraries

### Combined Advantages
✅ **Best of both worlds**: Compile-time speed + runtime flexibility
✅ **Unified heterogeneous computing**: CPU/GPU treated equally
✅ **Future-proof**: Easy to add FPGA, NPU, or new accelerators
✅ **Power-efficient**: Smart about device selection (battery vs AC power)

## Implementation Phases

### Phase 1: Complete Backend Traits ✅
1. ✅ Finish `gpu_backend_trait.h` (add missing 124 operations)
2. 🔄 Create `cpu_backend_trait.h` (mirror GPU trait + CPU-specific ops)
3. 🔄 Update all backend implementations (cuBLAS, rocBLAS, MKL, OpenBLAS)

### Phase 2: Compute Device Abstraction
1. Create `compute_device.h` with unified device interface
2. Implement device detection (query GPUs via CUDA/ROCm/SYCL, query CPU via cpuid)
3. Build operation cost models (calibration data per device)

### Phase 3: Smart Dispatch
1. Implement `compute_manager.h` with device selection logic
2. Add data tracking (optional, but valuable for avoiding transfers)
3. Create smart dispatch API (`fb_*_auto` functions)

### Phase 4: Policies and Optimization
1. Load balancing algorithm
2. Power-aware scheduling
3. Data locality optimization
4. Multi-operation pipelining (queue multiple ops, batch transfers)

## Recommendation: YES, Do Both!

**This is a superior architecture** that provides:
1. **Interchangeability**: Same 341 operations on CPU and GPU
2. **Performance**: Compile-time optimization within each backend
3. **Flexibility**: Runtime choice of device based on current conditions
4. **Intelligence**: System adapts to load, power, and data location

The hybrid approach makes faster-blaster **significantly more powerful** than existing solutions while maintaining zero-cost abstraction where it matters most (within each backend).
