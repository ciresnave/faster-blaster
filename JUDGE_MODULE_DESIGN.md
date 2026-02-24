# faster-blaster Judge Module: Design Specification

The judge module is the correctness and precision arbiter for faster-blaster's runtime
operation routing. It runs once at startup (or whenever a backend/device change is
detected) and produces precision profiles that feed the operation ranking tables used by
the router. It does **not** run during normal operation dispatch.

Source location: `src/judge/`

---

## 1. Purpose and lifecycle

```
startup / device change
        │
        ▼
  ┌─────────────────┐
  │  Judge module   │  Runs once per (op, backend, device, size_class, dtype)
  │                 │
  │  1. Generate    │◄── faster-blaster-reference oracle (via reference plugin vtable)
  │     test corpus │
  │  2. Exercise    │◄── candidate backend vtable
  │     candidate   │
  │  3. Compute     │
  │     precision   │──► fb_precision_profile_t (per-op, stored in judge_store)
  │     profiles    │
  └─────────────────┘
        │
        ▼
  ┌─────────────────┐
  │  Benchmark      │  Reads profiles to build routing tables
  │  cache          │  Stores compact summaries (guaranteed_digits, cost_model)
  └─────────────────┘
        │
        ▼
  ┌─────────────────┐
  │  Operation      │  Uses routing tables at dispatch time (zero judge overhead)
  │  router         │
  └─────────────────┘
```

The judge module is invisible to normal developer usage. Developers interact only with
the router, which consults pre-computed profiles from the benchmark cache.

---

## 2. Core data types

### 2.1 Precision profile: the accuracy curve

The fundamental insight is that "N digits correct" is not a binary property — it is a
distribution. An implementation may achieve 5 digits on 100% of inputs, 6 digits on 90%
of inputs, and 7 digits on 50% of inputs. The precision profile captures this curve so
the router can serve developer requests like "I need 6 digits with 80% reliability."

```c
// include/faster-blaster/judge.h

#define FB_JUDGE_MAX_CURVE_POINTS  16

// One point on the accuracy curve: "pass_rate fraction of test cases hit >= digits"
typedef struct {
    uint8_t  digits;       // number of significant decimal digits
    float    pass_rate;    // fraction of test cases achieving at least this many digits
                           // 1.0 = 100%, 0.5 = 50%, etc.
} fb_digits_point_t;

// Per-metric precision profile for one (op, backend, device, size_class, dtype)
typedef struct {
    fb_digits_point_t  curve[FB_JUDGE_MAX_CURVE_POINTS]; // descending by digits
    uint8_t            curve_len;           // number of valid points
    uint8_t            guaranteed_digits;   // digits where pass_rate == 1.0
    uint8_t            typical_digits;      // digits at pass_rate >= 0.50 (p50)
    uint8_t            max_observed_digits; // best digits seen on any test case
    bool               has_fatal_failure;   // true if any NaN/Inf/crash observed
    uint32_t           test_case_count;     // number of test cases run
} fb_metric_profile_t;

// Timing profile: produced in the same judge pass as the precision profile.
// These replace the separate benchmark_execute.c timing for ops the judge covers.
// Only standard (non-edge-case) corpus entries contribute to timing statistics.
// The first FB_JUDGE_WARMUP_RUNS calls per (backend, op, size_class) are discarded.
#define FB_JUDGE_WARMUP_RUNS  2

typedef struct {
    uint32_t  mean_ns;          // mean wall-clock time per op call (nanoseconds)
    uint32_t  p50_ns;           // median
    uint32_t  p95_ns;           // 95th percentile (captures occasional slow outliers)
    uint32_t  min_ns;           // best observed (steady-state throughput indicator)
    uint32_t  stddev_ns;
    uint32_t  sample_count;     // number of timed calls (after warmup, non-edge only)
} fb_timing_profile_t;

// Full precision + timing profile for one operation-backend-precision combination.
// The judge produces both in a single pass — timing is captured during the same
// candidate calls made for correctness comparison.
typedef struct {
    // Per-metric precision profiles (populated based on archetype)
    fb_metric_profile_t  values;        // eigenvalues, singular values (spectral ops)
    fb_metric_profile_t  direct;        // element-wise (DIRECT archetype)
    fb_metric_profile_t  reconstruction;// ||A - factors|| / ||A||
    fb_metric_profile_t  orthogonality; // ||Q^HQ - I|| / ||I||
    fb_metric_profile_t  residual;      // ||b - Ax|| / (||A||||x|| + ||b||) (solves)
    fb_metric_profile_t  pairs;         // eigenpair/triplet residuals (deep audit only)
    fb_metric_profile_t  subspace;      // subspace comparison (degenerate clusters only)

    // Timing profile (standard corpus entries only, warmup excluded)
    // For deep audit mode, only the candidate op call is timed — not the extra
    // residual computation calls that the judge adds on top.
    fb_timing_profile_t  timing;

    // Overall summary
    uint8_t              archetype;     // which fb_judge_archetype_t was applied
    fb_judge_limit_t     limiting_metric; // which metric determines overall score
    bool                 is_deep_audit;   // was deep audit (pairs) computed?

    // Oracle ceiling applied during this run
    uint8_t              oracle_max_certifiable; // from PRECISION_GUARANTEES.md table

    // Provenance
    uint32_t             op_id;
    uint32_t             backend_id;
    uint32_t             device_id;
    uint8_t              size_class;     // fb_size_class_t
    uint8_t              dtype;          // fb_precision_t
} fb_precision_profile_t;
```

### 2.2 Judge query: what a developer op-sequence requests

```c
// Specifies precision requirements for one operation slot in an op sequence
typedef struct {
    uint8_t           requested_digits;    // minimum digits required
    float             required_pass_rate;  // fraction of executions that must meet it
                                           // 1.0 = always, 0.8 = 80% of the time
    fb_judge_limit_t  required_metrics;    // which metrics must meet requested_digits
    bool              deep_audit;          // request eigenpair/triplet residuals
} fb_judge_query_t;

// Predefined metric policies (most developers use one of these)
#define FB_JUDGE_POLICY_ALL_PRIMARY \
    (FB_JUDGE_LIMIT_VALUES | FB_JUDGE_LIMIT_RECONSTRUCTION | FB_JUDGE_LIMIT_ORTHOGONALITY)

#define FB_JUDGE_POLICY_RECONSTRUCTION_ONLY \
    (FB_JUDGE_LIMIT_RECONSTRUCTION)

#define FB_JUDGE_POLICY_VALUES_ONLY \
    (FB_JUDGE_LIMIT_VALUES)

#define FB_JUDGE_POLICY_RESIDUAL_ONLY \
    (FB_JUDGE_LIMIT_RESIDUAL)
```

### 2.3 Querying a profile against a developer request

The router calls this to check whether a stored profile satisfies a developer's request:

```c
// Returns true if the profile meets the query's requirements.
// 'effective_digits_out' is the weakest digit score across required metrics
// at the requested pass_rate.
bool fb_profile_meets_query(
    const fb_precision_profile_t* profile,
    const fb_judge_query_t*       query,
    uint8_t*                      effective_digits_out  // nullable
);

// Walk the accuracy curve for a specific metric to find digits at a given pass_rate
uint8_t fb_metric_digits_at_rate(
    const fb_metric_profile_t* metric,
    float                       required_pass_rate
);
```

### 2.4 The limiting metric type

```c
typedef enum {
    FB_JUDGE_LIMIT_NONE              = 0,
    FB_JUDGE_LIMIT_VALUES            = 1 << 0,  // eigenvalues/singular values
    FB_JUDGE_LIMIT_DIRECT            = 1 << 1,  // element-wise output (GEMM etc.)
    FB_JUDGE_LIMIT_RECONSTRUCTION    = 1 << 2,  // factorization/spectral reconstruction
    FB_JUDGE_LIMIT_ORTHOGONALITY     = 1 << 3,  // Q orthogonality / U,V unitarity
    FB_JUDGE_LIMIT_RESIDUAL          = 1 << 4,  // solution residual (solves)
    FB_JUDGE_LIMIT_PAIRS             = 1 << 5,  // eigenpair/triplet residuals
    FB_JUDGE_LIMIT_SUBSPACE          = 1 << 6,  // degenerate subspace metric
    FB_JUDGE_LIMIT_NONCONVERGENCE    = 1 << 7,  // algorithm did not converge
    FB_JUDGE_LIMIT_NAN_INF           = 1 << 8,  // NaN or Inf in output
} fb_judge_limit_t;
```

---

## 3. Operation metadata registry

One entry per op_id (indexed by the same integer as MAX_OPERATIONS in benchmark_cache.c).

```c
typedef enum {
    FB_JUDGE_DIRECT        = 0,  // GEMM, AXPY, COPY — unique output
    FB_JUDGE_INDEX         = 1,  // ISAMAX — discrete, tie-aware
    FB_JUDGE_FACTORIZATION = 2,  // LU, QR, Cholesky — reconstruction residual
    FB_JUDGE_SOLVE         = 3,  // GESV, GELS — backward error + conditioning
    FB_JUDGE_SPECTRAL      = 4,  // SYEV, GESVD — subspace + reconstruction
} fb_judge_archetype_t;

typedef enum {
    FB_OUTPUT_UNIQUE    = 0,   // result is mathematically unique
    FB_OUTPUT_SIGN_FLIP = 1,   // vectors may have sign/phase ambiguity (well-separated)
    FB_OUTPUT_SUBSPACE  = 2,   // degenerate/clustered: only subspace is well-defined
    FB_OUTPUT_PIVOT     = 3,   // permutation ambiguity (equivalent factorizations)
    FB_OUTPUT_CONJUGATE_PAIR = 4, // real geev: complex conjugate pair ordering
    FB_OUTPUT_DISCRETE  = 5,   // integer index (argmax/argmin)
} fb_output_uniqueness_t;

typedef enum {
    FB_SCORE_MIN_PRIMARY         = 0,  // min(required metrics) determines overall score
    FB_SCORE_RECONSTRUCTION_ONLY = 1,  // only reconstruction matters
    FB_SCORE_VALUES_ONLY         = 2,  // only value accuracy matters (rare)
} fb_score_policy_t;

typedef struct {
    fb_judge_archetype_t    archetype;
    fb_output_uniqueness_t  uniqueness;
    fb_score_policy_t       default_score_policy;
    fb_judge_limit_t        primary_metrics;     // metrics computed in standard mode
    uint8_t                 max_certifiable_f32; // from PRECISION_GUARANTEES.md
    uint8_t                 max_certifiable_f64; // from PRECISION_GUARANTEES.md
    float                   ortho_slack_f32;     // allowed reduction vs value digits
    float                   ortho_slack_f64;
    double                  cluster_threshold_multiplier; // for spectral: n × eps
    bool                    is_in_place;         // input must be snapshotted before call
    const char*             name;
} fb_op_judge_meta_t;

extern const fb_op_judge_meta_t fb_op_judge_table[1248]; // in judge_metadata_table.c
```

---

## 4. Source file layout

```
src/judge/
    judge.h                 — public API (also in include/faster-blaster/)
    judge_types.h           — internal types shared across judge files
    judge_metadata.c        — registry lookup, cluster detection, oracle ceiling logic
    judge_metadata_table.c  — the static fb_op_judge_table[1248] entries
    judge_corpus.c          — test corpus generator per op family and size class
    judge_direct.c          — FB_JUDGE_DIRECT archetype implementation
    judge_index.c           — FB_JUDGE_INDEX archetype (tie-aware discrete comparison)
    judge_factorization.c   — FB_JUDGE_FACTORIZATION (reconstruction + pivot norm)
    judge_solve.c           — FB_JUDGE_SOLVE (residual + condition estimation)
    judge_spectral.c        — FB_JUDGE_SPECTRAL (values + recon + ortho + subspace)
    judge_profile.c         — fb_precision_profile_t accumulation and serialization
    judge_store.c           — on-disk profile cache (read/write profiles between runs)
    judge_norm.c            — shared norm utilities (Frobenius, spectral, vector norms)
    judge_condition.c       — condition number estimation utilities
```

---

## 5. Archetype implementations

### Phase 1: FB_JUDGE_DIRECT

Covers BLAS Level 1 (axpy, scal, copy, rot) and Level 3 (gemm, symm, trmm).
These have unique output — element-wise or matrix-wise comparison against reference.

```
digits = -log10( ||X - X*||_F / max(||X*||_F, tau) )
```

where tau = eps_type × input_scale_estimate (prevents log(0) near zero outputs).

Implementation in `judge_direct.c`. Feeds into `profile.direct`.

### Phase 2: FB_JUDGE_INDEX

Covers isamax, idamax, icamax, izamax and extension argmin operations.
Discrete output: check index match with reference, handle ties within eps of reference
maximum. See Section 2.13 of PRECISION_GUARANTEES.md for scoring rules.

Implementation in `judge_index.c`. No metric_profile curve — stored as pass/fail rate
plus `max_value_digits` (how many digits the maximum value itself agrees to).

### Phase 3: FB_JUDGE_FACTORIZATION

Covers getrf (LU), potrf (Cholesky), geqrf (QR), and their variants.
Reconstruction residual is the primary metric. Orthogonality for QR.

Safe-canonical operations applied before any comparison:
- Sort diagonal of L/U for display purposes only (do not modify for reconstruction).
- For QR: do not attempt to canonicalize Q columns; use reconstruction directly.

Implementation in `judge_factorization.c`. Feeds `profile.reconstruction` and
`profile.orthogonality`.

### Phase 4: FB_JUDGE_SOLVE

Covers gesv, posv, gels, getrs, potrs and all solve driver variants.
Backward error η = ||b - Ax|| / (||A||||x|| + ||b||) is the primary metric.
Condition number estimate recorded per test case.

Implementation in `judge_solve.c`. Feeds `profile.residual`.
Judge annotates each test case result with kappa_estimate so the router can filter
"physically impossible to certify" cases from "implementation error" cases.

### Phase 5: FB_JUDGE_SPECTRAL

Covers syev/heev, gesvd/gesdd, geev and all expert driver variants.
Most complex. Implemented last. See Section 2 of PRECISION_GUARANTEES.md for the
5-metric stack and safe/unsafe canonicalization rules.

Standard mode (fast, runs at every profiling run):
1. Sort values consistently (stable sort by value).
2. For real geev: canonicalize conjugate pairs to positive-imaginary-part-first.
3. Compute value accuracy digits.
4. Compute reconstruction residual.
5. Compute orthogonality (for unitary/orthogonal factor matrices).
6. Detect clusters (relative gap < cluster_threshold_multiplier × eps).
7. If clusters detected: compute subspace comparison for affected clusters.

Deep audit mode (slow, one-time backend certification):
8. Compute eigenpair / singular-triplet residuals.

Safe canonicalization only:
- Sign/phase alignment for well-separated vectors (relative gap > cluster threshold).
- This reduces comparison noise for debugging; it is not a correctness dependency.

Implementation in `judge_spectral.c`. Feeds `profile.values`, `profile.reconstruction`,
`profile.orthogonality`, `profile.subspace` (conditional), `profile.pairs` (deep audit).

---

## 6. Oracle failure handling policy

If the faster-blaster-reference oracle returns NaN, Inf, or does not converge on a test
case, the judge must:

1. **Log the failure immediately** with full input parameters (op_id, dtype, dimensions,
   seed, input values).
2. **Classify the cause**:
   - If the input is mathematically invalid (e.g., non-positive-definite matrix for
     Cholesky): discard the test case from the corpus, log it as "invalid input."
   - If the input is mathematically valid and the oracle still fails: **this is a bug
     in faster-blaster-reference**. The judge must halt profiling for this operation,
     emit `FB_STATUS_ORACLE_FAILURE`, and require the reference bug to be fixed before
     profiling can resume. The reference is the single source of truth; if it is wrong,
     it must be corrected before any candidate can be evaluated.
3. **Never** count an oracle failure as a candidate failure.

---

## 7. Test corpus design

The corpus generator (`judge_corpus.c`) produces test cases per (op, dtype, size_class):

| Category                | Description                                  | kappa range  |
| ----------------------- | -------------------------------------------- | ------------ |
| Normal random           | Random matrix, no structure                  | O(n)         |
| Well-conditioned        | Synthetic with κ ≤ 10                        | 1 – 10       |
| Moderately conditioned  | κ ≈ 10³ – 10⁶                                | 10³ – 10⁶    |
| Near-singular           | κ approaching 1/eps                          | 10¹² – 1/eps |
| Structured (SPD, tri.)  | Satisfies the domain precondition exactly    | varies       |
| Edge: extreme scale     | Elements spanning 10 orders of magnitude     | varies       |
| Edge: tiny/zero columns | Rank-deficient or near-rank-deficient        | ∞ or large   |
| Degenerate spectral     | Repeated eigenvalues/singular values for eq. | varies       |

Corpus entries carry:
```c
typedef struct {
    uint32_t  op_id;
    uint8_t   dtype;
    uint8_t   size_class;
    uint64_t  seed;          // deterministic reproduction
    double    kappa_estimate; // -1.0 if not applicable
    bool      is_degenerate_spectrum; // triggers subspace judge path
    bool      is_edge_case;           // weighted differently in profile statistics
} fb_corpus_entry_t;
```

Edge cases are included for completeness but are **down-weighted** when computing
`guaranteed_digits` and `pass_rate` statistics. The `is_edge_case` flag lets the profile
report two digit curves: "standard inputs" and "including edge cases."

---

## 8. Profile storage and invalidation

Profiles are persisted to `<build_dir>/judge_profiles/` as binary blobs, one file per
`(op_id, backend_id, device_id, dtype)`. The judge re-runs for a given entry if:

- The backend shared library has a newer modification timestamp.
- The device hardware ID has changed (new GPU detected or removed).
- The corpus version has changed (new test cases added).
- The judge module version has incremented (algorithm improvements).

A manifest file (`judge_manifest.json`) tracks versions. The router reads from the
manifest at startup to determine whether profiles are current.

---

## 9. Implementation phases

| Phase | Module(s)                                   | Archetype covered      |
| ----- | ------------------------------------------- | ---------------------- |
| 1     | judge_types.h, judge_norm.c, judge_direct.c | FB_JUDGE_DIRECT        |
|       | judge_corpus.c (normal + edge scale cases)  |                        |
|       | judge_profile.c, judge_metadata_table.c     |                        |
| 2     | judge_index.c                               | FB_JUDGE_INDEX         |
| 3     | judge_factorization.c, judge_condition.c    | FB_JUDGE_FACTORIZATION |
|       | judge_corpus.c (structured + near-singular) |                        |
| 4     | judge_solve.c                               | FB_JUDGE_SOLVE         |
| 5     | judge_spectral.c                            | FB_JUDGE_SPECTRAL      |
|       | judge_corpus.c (degenerate spectrum cases)  |                        |
| 6     | judge_store.c, judge_metadata.c (full)      | All — persistence      |

Phase 1 unblocks BLAS Level 1 and Level 3 ranking immediately. Each subsequent phase
unlocks ranking for its operation family. The router can use Phase 1 results while
Phases 2–6 complete.
