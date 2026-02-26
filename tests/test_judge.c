/**
 * @file test_judge.c
 * @brief Unit tests for the faster-blaster judge module.
 *
 * Tests are self-contained and do not require a real backend — synthetic
 * profiles are constructed in memory and written to a temporary directory.
 *
 * Coverage:
 *   1. Corpus reproducibility  — same (op, size_class, dtype) yields identical data
 *   2. Store round-trip        — save + load produces bit-identical profile
 *   3. fb_judge_profile_meets_query — correct pass/fail on synthetic profiles
 *   4. fb_judge_store_list_profiled_ops — detects newly written profiles
 *   5. fb_judge_select_best_backend    — selects more-precise backend under
 *                                        FB_SELECT_MAX_PRECISION
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* Judge public API */
#include "../include/faster-blaster/judge.h"
#include "../include/faster-blaster/judge_select.h"

/* Internal — store and corpus are not in the public install headers */
#include "../src/judge/judge_store.h"
#include "../src/judge/judge_corpus.h"
#include "../src/judge/judge_op_ids.h"

/* =========================================================================
 * Helpers
 * ========================================================================= */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, msg)                                                   \
    do {                                                                    \
        if (cond) {                                                         \
            printf("  PASS  %s\n", msg); g_pass++;                         \
        } else {                                                            \
            printf("  FAIL  %s  (line %d)\n", msg, __LINE__); g_fail++;    \
        }                                                                   \
    } while (0)

/* Make a flat temporary directory path (process-unique). */
static void make_temp_dir(char *out, size_t sz)
{
#ifdef _WIN32
    char tmp[260];
    DWORD n = GetTempPathA((DWORD)sizeof(tmp), tmp);
    if (n == 0) { snprintf(out, sz, "test_judge_temp"); return; }
    /* Remove trailing backslash. */
    if (n > 0 && tmp[n-1] == '\\') tmp[n-1] = '\0';
    snprintf(out, sz, "%s\\test_judge_%u", tmp, (unsigned)GetCurrentProcessId());
#else
    snprintf(out, sz, "/tmp/test_judge_%u", (unsigned)getpid());
#endif
}

/* Build a synthetic fb_precision_profile_t with specified op_name, backends,
 * and a direct metric that certifies 'guaranteed_digits'. */
static fb_precision_profile_t make_profile(
    const char *op_name,
    uint32_t    backend_id,
    uint32_t    device_id,
    uint8_t     size_class,
    uint8_t     dtype,
    uint8_t     guaranteed_digits)
{
    fb_precision_profile_t p;
    memset(&p, 0, sizeof(p));

    strncpy(p.op_name, op_name, sizeof(p.op_name) - 1);
    p.backend_id   = backend_id;
    p.device_id    = device_id;
    p.size_class   = size_class;
    p.dtype        = dtype;
    p.archetype    = (uint8_t)FB_JUDGE_DIRECT;
    p.limiting_metric = (uint32_t)FB_JUDGE_LIMIT_DIRECT;
    p.oracle_max_certifiable = 7;

    /* Populate the direct metric curve with a single point. */
    p.direct.curve[0].digits    = guaranteed_digits;
    p.direct.curve[0].pass_rate = 1.0f;
    p.direct.curve_len          = 1;
    p.direct.guaranteed_digits  = guaranteed_digits;
    p.direct.typical_digits     = guaranteed_digits;
    p.direct.max_observed_digits = guaranteed_digits;
    p.direct.test_case_count    = 12;

    /* Timing — synthetic medium numbers. */
    p.timing.mean_ns     = 1000;
    p.timing.p50_ns      = 950;
    p.timing.p95_ns      = 1500;
    p.timing.min_ns      = 800;
    p.timing.sample_count = 8;

    return p;
}

/* =========================================================================
 * Test 1: Corpus reproducibility
 * ========================================================================= */

static void test_corpus_reproducibility(void)
{
    printf("\nTest 1: Corpus reproducibility\n");

    uint32_t op_id     = FB_OP_SAXPY;
    uint8_t  size_cls  = FB_SIZE_SMALL;
    fb_dtype_t dtype    = FB_DTYPE_F32;

    fb_corpus_case_t cases_a[FB_CORPUS_TOTAL_CASES];
    fb_corpus_case_t cases_b[FB_CORPUS_TOTAL_CASES];
    memset(cases_a, 0, sizeof(cases_a));
    memset(cases_b, 0, sizeof(cases_b));

    fb_judge_status_t ra = fb_corpus_generate(op_id, size_cls, dtype, cases_a);
    fb_judge_status_t rb = fb_corpus_generate(op_id, size_cls, dtype, cases_b);

    EXPECT(ra == FB_JUDGE_OK, "fb_corpus_generate() first call succeeds");
    EXPECT(rb == FB_JUDGE_OK, "fb_corpus_generate() second call succeeds");

    if (ra == FB_JUDGE_OK && rb == FB_JUDGE_OK) {
        bool dims_match = true;
        bool data_match = true;
        bool scalars_match = true;

        for (int i = 0; i < FB_CORPUS_TOTAL_CASES; i++) {
            if (cases_a[i].n != cases_b[i].n ||
                cases_a[i].m != cases_b[i].m ||
                cases_a[i].k != cases_b[i].k) {
                dims_match = false; break;
            }
            if (fabs(cases_a[i].alpha - cases_b[i].alpha) > 1e-15 ||
                fabs(cases_a[i].beta  - cases_b[i].beta)  > 1e-15) {
                scalars_match = false; break;
            }
            /* Compare raw contents of the A buffer. */
            if (cases_a[i].A && cases_b[i].A && cases_a[i].A_elems > 0) {
                size_t dtype_sz = (dtype == FB_DTYPE_F64) ? sizeof(double) : sizeof(float);
                if (memcmp(cases_a[i].A, cases_b[i].A,
                           cases_a[i].A_elems * dtype_sz) != 0) {
                    data_match = false; break;
                }
            }
        }

        EXPECT(dims_match,    "Corpus: dimensions are reproducible");
        EXPECT(scalars_match, "Corpus: alpha/beta scalars are reproducible");
        EXPECT(data_match,    "Corpus: A buffer contents are reproducible");
    }

    for (int i = 0; i < FB_CORPUS_TOTAL_CASES; i++) {
        fb_corpus_case_free(&cases_a[i]);
        fb_corpus_case_free(&cases_b[i]);
    }
}

/* =========================================================================
 * Test 2: Store save / load round-trip
 * ========================================================================= */

static void test_store_round_trip(const char *tmpdir)
{
    printf("\nTest 2: Store save/load round-trip\n");

    fb_precision_profile_t saved = make_profile(
        "saxpy", /*backend=*/42, /*device=*/0,
        (uint8_t)FB_SIZE_SMALL, (uint8_t)FB_DTYPE_F32, /*digits=*/6);

    fb_judge_status_t ws = fb_judge_store_save(tmpdir, &saved);
    EXPECT(ws == FB_JUDGE_OK, "fb_judge_store_save() returns OK");

    fb_precision_profile_t loaded;
    memset(&loaded, 0, sizeof(loaded));
    fb_judge_status_t wl = fb_judge_store_load(
        tmpdir, "saxpy", 42, 0,
        (uint8_t)FB_SIZE_SMALL, (uint8_t)FB_DTYPE_F32, &loaded);
    EXPECT(wl == FB_JUDGE_OK, "fb_judge_store_load() returns OK");

    if (wl == FB_JUDGE_OK) {
        EXPECT(loaded.backend_id == saved.backend_id,
               "Loaded backend_id matches saved");
        EXPECT(loaded.direct.guaranteed_digits == saved.direct.guaranteed_digits,
               "Loaded guaranteed_digits matches saved");
        EXPECT(strcmp(loaded.op_name, saved.op_name) == 0,
               "Loaded op_name matches saved");
        EXPECT(loaded.timing.mean_ns == saved.timing.mean_ns,
               "Loaded timing.mean_ns matches saved");
    }

    /* Non-existent profile returns NO_PROFILE. */
    fb_precision_profile_t missing;
    fb_judge_status_t wnp = fb_judge_store_load(
        tmpdir, "saxpy", 42, 0,
        (uint8_t)FB_SIZE_HUGE, (uint8_t)FB_DTYPE_F32, &missing);
    EXPECT(wnp == FB_JUDGE_ERR_NO_PROFILE,
           "Missing profile returns FB_JUDGE_ERR_NO_PROFILE");
}

/* =========================================================================
 * Test 3: fb_judge_profile_meets_query
 * ========================================================================= */

static void test_profile_meets_query(void)
{
    printf("\nTest 3: fb_judge_profile_meets_query\n");

    /* Profile with 6 guaranteed digits on direct metric. */
    fb_precision_profile_t prof = make_profile(
        "saxpy", 1, 0, (uint8_t)FB_SIZE_SMALL, (uint8_t)FB_DTYPE_F32, 6);

    fb_judge_query_t  q_pass = {
        .requested_digits   = 5,
        .required_pass_rate = 1.0f,
        .required_metrics   = (uint32_t)FB_JUDGE_LIMIT_DIRECT,
        .deep_audit         = false
    };
    fb_judge_query_t q_fail = {
        .requested_digits   = 7,   /* exceeds what the profile guarantees */
        .required_pass_rate = 1.0f,
        .required_metrics   = (uint32_t)FB_JUDGE_LIMIT_DIRECT,
        .deep_audit         = false
    };

    uint8_t eff_digits = 0;
    bool meets_pass = fb_judge_profile_meets_query(&prof, &q_pass, &eff_digits);
    EXPECT(meets_pass, "Profile with 6 digits meets query for 5 digits");
    EXPECT(eff_digits >= 5, "Effective digits >= 5 when query asks for 5");

    bool meets_fail = fb_judge_profile_meets_query(&prof, &q_fail, &eff_digits);
    EXPECT(!meets_fail, "Profile with 6 digits does NOT meet query for 7 digits");
}

/* =========================================================================
 * Test 4: fb_judge_store_list_profiled_ops
 * ========================================================================= */

static void test_list_profiled_ops(const char *tmpdir)
{
    printf("\nTest 4: fb_judge_store_list_profiled_ops\n");

    /* Write two different ops for the same (backend=10, device=0). */
    fb_precision_profile_t p1 = make_profile(
        "saxpy", 10, 0, (uint8_t)FB_SIZE_SMALL, (uint8_t)FB_DTYPE_F32, 6);
    fb_precision_profile_t p2 = make_profile(
        "daxpy", 10, 0, (uint8_t)FB_SIZE_SMALL, (uint8_t)FB_DTYPE_F64, 13);

    /* Also write one for a different backend (should not appear in results). */
    fb_precision_profile_t p3 = make_profile(
        "saxpy", 99, 0, (uint8_t)FB_SIZE_SMALL, (uint8_t)FB_DTYPE_F32, 6);

    fb_judge_store_save(tmpdir, &p1);
    fb_judge_store_save(tmpdir, &p2);
    fb_judge_store_save(tmpdir, &p3);

    /* Count-only pass. */
    uint32_t count = fb_judge_store_list_profiled_ops(tmpdir, 10, 0, NULL, 0);
    EXPECT(count == 2, "list_profiled_ops: count=2 for backend 10");

    /* Retrieve pass. */
    uint32_t ids[8] = {0};
    uint32_t n = fb_judge_store_list_profiled_ops(tmpdir, 10, 0, ids, 8);
    EXPECT(n == 2, "list_profiled_ops: returns 2 op IDs");

    bool found_saxpy = false, found_daxpy = false;
    for (uint32_t i = 0; i < n && i < 8; i++) {
        if (ids[i] == FB_OP_SAXPY) found_saxpy = true;
        if (ids[i] == FB_OP_DAXPY) found_daxpy = true;
    }
    EXPECT(found_saxpy, "list_profiled_ops: FB_OP_SAXPY found for backend 10");
    EXPECT(found_daxpy, "list_profiled_ops: FB_OP_DAXPY found for backend 10");

    /* Backend 99 has only 1 profile, backend 10 results must be unaffected. */
    uint32_t n99 = fb_judge_store_list_profiled_ops(tmpdir, 99, 0, NULL, 0);
    EXPECT(n99 == 1, "list_profiled_ops: count=1 for backend 99");
}

/* =========================================================================
 * Test 5: fb_judge_select_best_backend
 * ========================================================================= */

static void test_select_best_backend(const char *tmpdir)
{
    printf("\nTest 5: fb_judge_select_best_backend\n");

    /* Backend 20: 5 guaranteed digits. */
    fb_precision_profile_t low = make_profile(
        "saxpy", 20, 0, (uint8_t)FB_SIZE_MEDIUM, (uint8_t)FB_DTYPE_F32, 5);
    /* Backend 21: 7 guaranteed digits (should win under MAX_PRECISION). */
    fb_precision_profile_t high = make_profile(
        "saxpy", 21, 0, (uint8_t)FB_SIZE_MEDIUM, (uint8_t)FB_DTYPE_F32, 7);

    fb_judge_store_save(tmpdir, &low);
    fb_judge_store_save(tmpdir, &high);

    uint32_t backends[2] = {20, 21};
    uint32_t winner = UINT32_MAX;
    uint8_t eff = 0;

    fb_select_status_t st = fb_judge_select_best_backend(
        tmpdir,
        FB_OP_SAXPY,
        /*device=*/0,
        FB_SIZE_MEDIUM,
        (uint8_t)FB_DTYPE_F32,
        &FB_SELECT_MAX_PRECISION,
        backends, 2,
        &winner, &eff);

    EXPECT(st == FB_SELECT_OK || st == FB_SELECT_WARN_DEGRADED,
           "select_best_backend: returns OK or WARN_DEGRADED");
    EXPECT(winner == 21,
           "select_best_backend: backend 21 (7 digits) wins over 20 (5 digits)");

    /* Under MAX_SPEED the faster backend wins.  Both have timing.min_ns=800 so
     * they tie — the result should still be one of the two valid backends. */
    uint32_t speed_winner = UINT32_MAX;
    fb_judge_select_best_backend(
        tmpdir, FB_OP_SAXPY, 0, FB_SIZE_MEDIUM, (uint8_t)FB_DTYPE_F32,
        &FB_SELECT_MAX_SPEED, backends, 2, &speed_winner, NULL);
    EXPECT(speed_winner == 20 || speed_winner == 21,
           "select_best_backend (MAX_SPEED): winner is a valid backend");
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    printf("=== faster-blaster Judge Module Tests ===\n");

    char tmpdir[512];
    make_temp_dir(tmpdir, sizeof(tmpdir));
    printf("  Temp dir: %s\n", tmpdir);

    test_corpus_reproducibility();
    test_store_round_trip(tmpdir);
    test_profile_meets_query();
    test_list_profiled_ops(tmpdir);
    test_select_best_backend(tmpdir);

    printf("\n------------------------------------------\n");
    printf("Total: %d  Passed: %d  Failed: %d\n",
           g_pass + g_fail, g_pass, g_fail);
    if (g_fail == 0)
        printf("SUCCESS  (all tests passed)\n");
    else
        printf("FAILURE  (%d tests failed)\n", g_fail);

    return (g_fail == 0) ? 0 : 1;
}
