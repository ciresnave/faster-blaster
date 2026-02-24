# Faster-BLASTER Community Benchmark Database

**Public benchmark database for BLAS/LAPACK operations across all hardware and backends.**

> 🚀 Help the community! Submit your benchmark results and see how your hardware ranks!

## 🎯 Project Goals

1. **Hardware Leaderboards** - See which GPUs/CPUs dominate each operation
2. **Backend Comparison** - Compare cuBLAS vs rocBLAS vs oneMKL on identical hardware
3. **Backend Vendor Accountability** - Vendors compete for best performance/accuracy
4. **Developer Intelligence** - Choose backends based on real-world data
5. **Faster-BLASTER Marketing** - Show that multi-backend dispatch wins!

## 📊 Features

### Leaderboards
- **Top Hardware by Operation** - "Fastest SGEMM: RTX 4090 (12,345 GFLOPS)"
- **Top Backends by Hardware** - "Best backend on Ryzen 9 7940HX: AOCL"
- **Most Accurate Implementations** - "Highest precision: Intel MKL"
- **Power Efficiency Rankings** - "Best GFLOPS/watt: Apple M3 Max"

### Comparison Tools
- **Hardware Comparison** - "RTX 4090 vs RX 7900 XTX across all operations"
- **Backend Comparison** - "cuBLAS 13.0 vs 12.5 performance delta"
- **System Comparison** - "Desktop vs Mobile GPU performance"

### Visualization
- **Interactive Charts** - Operation performance across hardware
- **Heatmaps** - Backend × Hardware performance matrix
- **Trend Analysis** - Performance improvements over driver versions
- **Accuracy vs Speed** - Pareto frontier for each operation

## ✅ Benchmark Integrity Policy

To keep rankings trustworthy and actionable, faster-blaster uses **bucketed benchmarking** and a **reference-oracle policy**:

- **Bucketed results**: Rankings are **per size/shape bucket**, not global winners.
- **Warm-up required**: Each benchmark includes warm-up iterations before timing.
- **Statistical sampling**: Mean, standard deviation, and p99 latency are recorded.
- **Noise control**: High-variance runs are re-sampled or flagged.

### Reference Backend Policy

- The reference backend is a **correctness oracle**, included in runs for validation.
- It may **win only within the specific bucket where it is fastest** (usually tiny sizes).
- It is **not** a global default and does **not** override optimized backends for larger sizes.
- If reference wins outside tiny buckets, treat it as a signal to re-check benchmark configuration, compiler flags, or vectorization.

## 🔧 Submission Process

### Automatic Submission from faster-blaster
```c
// In user code (opt-in)
fb_init_config_t config = {
    .enable_telemetry = true,          // Opt-in to submissions
    .submit_benchmarks = true,          // Auto-submit results
    .anonymous = true                   // No identifying info
};
fb_init(&config);
```

### Manual Submission
```bash
# Export benchmark results
fb-benchmark --export my_benchmarks.json

# Submit to database (requires GitHub account for PR)
fb-benchmark --submit my_benchmarks.json
```

### Data Validation (GitHub Actions)
- **Schema validation** - Ensure JSON matches schema
- **Sanity checks** - Flag impossible results (e.g., SGEMM > 100 TFLOPS)
- **Duplicate detection** - Merge results from same hardware
- **Verification status** - "Verified" badge for reproducible results

## 🏆 Vendor Bragging Rights

### Backend Vendor Pages
- **NVIDIA cuBLAS** - Showcase performance on their hardware
- **AMD rocBLAS** - Highlight Zen optimizations
- **Intel oneMKL** - Demonstrate AVX-512 advantages

### Marketing Opportunities
- "cuBLAS achieves 15 TFLOPS on RTX 4090" (link to leaderboard)
- "rocBLAS 30% faster than OpenBLAS on Ryzen 9950X"
- "faster-blaster automatically selects best: 2.3x speedup vs single backend"

## 🎨 Web Frontend

### Technology Stack
- **Frontend**: Vue.js or React
- **Charts**: Chart.js or D3.js
- **Hosting**: GitHub Pages (free!)
- **API**: Static JSON (no server needed initially)
- **Search**: Client-side Fuse.js

### Pages
1. **Home** - Overview, recent submissions, top performers
2. **Leaderboards** - Filter by operation, hardware, backend
3. **Hardware Browser** - Explore all tested hardware
4. **Backend Browser** - Explore all backends
5. **Compare** - Side-by-side comparison tool
6. **Submit** - Upload benchmark results
7. **About** - Project info, submission guidelines

## 📈 Growth Strategy

### Phase 1: Launch (Months 1-3)
- Basic web frontend
- Manual submissions via PR
- 10-20 hardware configurations

### Phase 2: Growth (Months 4-6)
- Auto-submit from faster-blaster
- Vendor outreach (NVIDIA, AMD, Intel)
- Academic partnerships (universities with GPUs)

### Phase 3: Scale (Months 7-12)
- 100+ hardware configurations
- Backend vendor pages
- Integration with HuggingFace, PyTorch, TensorFlow communities

## 🔐 Privacy & Ethics

### Data Collection
- **Opt-in only** - No automatic submissions without permission
- **Anonymous by default** - Hardware info only, no user data
- **Minimal data** - Only benchmark results, no system info beyond hardware
- **Verification** - Results marked as "verified" if reproducible

### Usage Guidelines
- Results used for performance comparison only
- No warranty on accuracy
- Community-driven, no official endorsements
- Open data (CC0 or MIT license)

## 💡 Benefits to faster-blaster

### Marketing
- **Proof of Concept** - "faster-blaster achieves 2-3x speedup vs single backend"
- **Community Engagement** - Users submit benchmarks → invested in project
- **Visibility** - Referenced by backend vendors, academic papers
- **Credibility** - Large benchmark database → serious project

### Technical
- **Pre-computed Benchmarks** - Users download instead of running locally
- **Hardware Recommendations** - "Your RTX 4070 works best with cuBLAS"
- **Performance Regression Detection** - Track driver/backend regressions
- **Validation** - Community can verify correctness claims

## 🚀 Implementation Plan

### Immediate (Next 2 weeks)
1. Create `faster-blaster/benchmark-database` repository
2. Define JSON schema
3. Manual submission via PR (no web frontend yet)
4. Seed with 5-10 initial hardware configs

### Short-term (Next 1-2 months)
1. Basic web frontend (GitHub Pages)
2. Leaderboard implementation
3. Auto-submit from faster-blaster
4. Vendor outreach

### Long-term (3-6 months)
1. Advanced visualization
2. Comparison tools
3. Vendor pages
4. Academic partnerships
5. Integration with ML frameworks

## 🤝 Call to Action

**This creates a virtuous cycle:**
1. Users submit benchmarks → Database grows
2. Database grows → More valuable to community
3. More valuable → More users adopt faster-blaster
4. More users → More benchmarks submitted
5. Vendors compete → Better backends for everyone!

**Let's build the "PassMark for BLAS/LAPACK"!** 🎯
