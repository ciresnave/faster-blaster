/**
 * @file test_ranked_dispatch.c
 * @brief Unit tests for the Phase 3 ranked dispatch table API.
 *
 * Tests are self-contained.  No real backend profiles are required:
 * most tests exercise the empty-table and availability machinery; the
 * save/load round-trip test writes to a temporary file.
 *
 * Coverage:
 *   1. Build with non-existent profile dir   — returns valid empty table
 *   2. get_best on empty table               — returns fallback_backend_id
 *   3. get_with_floor on empty table         — returns fallback
 *   4. entry_count on empty table            — returns 0
 *   5. get_entry out-of-range               — returns NULL
 *   6. get_stats on empty table              — all fields zero
 *   7. mark_unavailable / is_available       — round-trip
 *   8. mark_unavailable + mark_available     — cancel each other
 *   9. All criteria axes covered by get_best — still returns fallback
 *  10. save / load round-trip               — reloaded table matches original
 *  11. Unavailability survives save/load     — persisted in binary file
 *  12. NULL-safety: NULL table never crashes — all API functions tolerate NULL
 *  13. n_backends capped at FB_RANKED_MAX_BACKENDS — build does not crash
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "../include/faster-blaster/ranked_dispatch.h"
#include "../include/faster-blaster/backend_ids.h"
#include "../src/judge/judge_op_ids.h"   /* FB_OP_SAXPY */

/* =========================================================================
 * Minimal test framework (mirrors test_judge.c pattern)
 * ========================================================================= */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, msg)                                                    \
    do {                                                                     \
        if (cond) {                                                          \
            printf("  PASS  %s\n", (msg));  g_pass++;                       \
        } else {                                                             \
            printf("  FAIL  %s  (line %d)\n", (msg), __LINE__); g_fail++;   \
        }                                                                    \
    } while (0)

/* Temporary file path for save/load tests. */
static void temp_path(char *out, size_t sz, const char *name)
{
    const char *base = getenv("TEMP");
    if (!base) base  = getenv("TMPDIR");
    if (!base) base  = "/tmp";
    snprintf(out, sz, "%s/%s", base, name);
}

/* =========================================================================
 * Shared test fixtures
 * ========================================================================= */

/* Three real-ish backend IDs used across tests. */
static const uint32_t k_backends[] = {
    FB_BACKEND_ID_AOCL_BLIS,
    FB_BACKEND_ID_MKL,
    FB_BACKEND_ID_OPENBLAS,
};
static const uint32_t k_n_backends = 3u;
static const uint32_t k_fallback   = FB_BACKEND_ID_REFERENCE;

/* Non-existent profile dir so the table builds as empty. */
static const char *k_empty_dir = "C:/nonexistent-fb-profiles-zzz";

/* =========================================================================
 * Test 1: Build with non-existent profile dir
 * ========================================================================= */

static void test_build_empty(void)
{
    printf("\nTest 1: build with non-existent profile dir\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0 /* FB_DTYPE_F32 */,
        k_backends, k_n_backends, k_fallback);

    EXPECT(tbl != NULL, "table is non-NULL even when no profiles found");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 2: get_best on empty table
 * ========================================================================= */

static void test_get_best_empty(void)
{
    printf("\nTest 2: get_best on empty table returns fallback\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!tbl) { printf("  SKIP  (table alloc failed)\n"); return; }

    uint32_t b;

    b = fb_ranked_get_best(tbl, FB_OP_SAXPY, FB_RANK_FASTEST);
    EXPECT(b == k_fallback, "FB_RANK_FASTEST returns fallback for saxpy");

    b = fb_ranked_get_best(tbl, FB_OP_SAXPY, FB_RANK_MOST_ACCURATE);
    EXPECT(b == k_fallback, "FB_RANK_MOST_ACCURATE returns fallback for saxpy");

    b = fb_ranked_get_best(tbl, FB_OP_SAXPY, FB_RANK_BALANCED);
    EXPECT(b == k_fallback, "FB_RANK_BALANCED returns fallback for saxpy");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 3: get_with_floor on empty table
 * ========================================================================= */

static void test_get_with_floor_empty(void)
{
    printf("\nTest 3: get_with_floor on empty table returns fallback\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!tbl) { printf("  SKIP  (table alloc failed)\n"); return; }

    uint32_t b;

    /* min_digits=0 means no constraint — should still return fallback. */
    b = fb_ranked_get_with_floor(tbl, FB_OP_SAXPY, FB_RANK_FASTEST, 0);
    EXPECT(b == k_fallback, "floor=0 on empty returns fallback");

    /* min_digits=15 — no backend can satisfy, returns fallback. */
    b = fb_ranked_get_with_floor(tbl, FB_OP_SAXPY, FB_RANK_FASTEST, 15);
    EXPECT(b == k_fallback, "floor=15 on empty returns fallback");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 4: entry_count on empty table
 * ========================================================================= */

static void test_entry_count_empty(void)
{
    printf("\nTest 4: entry_count on empty table returns 0\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!tbl) { printf("  SKIP  (table alloc failed)\n"); return; }

    uint32_t cnt;

    cnt = fb_ranked_entry_count(tbl, FB_OP_SAXPY, FB_RANK_FASTEST);
    EXPECT(cnt == 0u, "entry_count FASTEST for saxpy is 0 (no profiles)");

    cnt = fb_ranked_entry_count(tbl, FB_OP_SAXPY, FB_RANK_MOST_ACCURATE);
    EXPECT(cnt == 0u, "entry_count MOST_ACCURATE for saxpy is 0");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 5: get_entry out-of-range
 * ========================================================================= */

static void test_get_entry_oob(void)
{
    printf("\nTest 5: get_entry out-of-range returns NULL\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!tbl) { printf("  SKIP  (table alloc failed)\n"); return; }

    const fb_ranked_entry_t *e;

    /* rank 0 on empty table — no valid entries. */
    e = fb_ranked_get_entry(tbl, FB_OP_SAXPY, FB_RANK_FASTEST, 0u);
    EXPECT(e == NULL, "get_entry rank=0 on empty returns NULL");

    /* rank beyond FB_RANKED_TOP_N — always NULL. */
    e = fb_ranked_get_entry(tbl, FB_OP_SAXPY, FB_RANK_FASTEST,
                            FB_RANKED_TOP_N + 1u);
    EXPECT(e == NULL, "get_entry rank > TOP_N returns NULL");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 6: get_stats on empty table
 * ========================================================================= */

static void test_stats_empty(void)
{
    printf("\nTest 6: get_stats on empty table reports zeros\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!tbl) { printf("  SKIP  (table alloc failed)\n"); return; }

    fb_ranked_stats_t st;
    memset(&st, 0xAA, sizeof(st));   /* poison */
    fb_ranked_table_get_stats(tbl, &st);

    EXPECT(st.ops_with_profiles == 0u, "ops_with_profiles == 0");
    EXPECT(st.ops_fully_ranked  == 0u, "ops_fully_ranked == 0");
    EXPECT(st.total_entries     == 0u, "total_entries == 0");
    /* distinct_winners may be 0 or undefined on empty — just check struct
     * was written (not poisoned 0xAAAAAAAA). */
    EXPECT(st.distinct_winners < 0xAAAAAAAAu, "distinct_winners written");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 7: mark_unavailable / is_available round-trip
 * ========================================================================= */

static void test_availability_roundtrip(void)
{
    printf("\nTest 7: mark_unavailable / is_available round-trip\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!tbl) { printf("  SKIP  (table alloc failed)\n"); return; }

    /* All backends should start available. */
    EXPECT(fb_ranked_is_available(tbl, FB_BACKEND_ID_MKL),
           "MKL starts available");
    EXPECT(fb_ranked_is_available(tbl, FB_BACKEND_ID_OPENBLAS),
           "OpenBLAS starts available");

    /* Mark MKL unavailable. */
    fb_ranked_mark_unavailable(tbl, FB_BACKEND_ID_MKL);
    EXPECT(!fb_ranked_is_available(tbl, FB_BACKEND_ID_MKL),
           "MKL unavailable after mark_unavailable");
    EXPECT(fb_ranked_is_available(tbl, FB_BACKEND_ID_OPENBLAS),
           "OpenBLAS still available after MKL marked");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 8: mark_unavailable + mark_available cancel
 * ========================================================================= */

static void test_availability_cancel(void)
{
    printf("\nTest 8: mark_available restores availability\n");

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!tbl) { printf("  SKIP  (table alloc failed)\n"); return; }

    fb_ranked_mark_unavailable(tbl, FB_BACKEND_ID_AOCL_BLIS);
    EXPECT(!fb_ranked_is_available(tbl, FB_BACKEND_ID_AOCL_BLIS),
           "AOCL_BLIS unavailable after mark_unavailable");

    fb_ranked_mark_available(tbl, FB_BACKEND_ID_AOCL_BLIS);
    EXPECT(fb_ranked_is_available(tbl, FB_BACKEND_ID_AOCL_BLIS),
           "AOCL_BLIS available again after mark_available");

    fb_ranked_table_free(tbl);
}

/* =========================================================================
 * Test 9: build with FB_RANKED_MAX_BACKENDS + 4 — no crash
 * ========================================================================= */

static void test_n_backends_cap(void)
{
    printf("\nTest 9: n_backends > FB_RANKED_MAX_BACKENDS caps safely\n");

    uint32_t ids[FB_RANKED_MAX_BACKENDS + 4];
    for (uint32_t i = 0; i < FB_RANKED_MAX_BACKENDS + 4; ++i)
        ids[i] = i;

    fb_ranked_table_t *tbl = fb_ranked_table_build(
        k_empty_dir, 0, 0,
        ids, FB_RANKED_MAX_BACKENDS + 4u, k_fallback);

    EXPECT(tbl != NULL, "build with oversized backend list does not return NULL");

    if (tbl) {
        uint32_t b = fb_ranked_get_best(tbl, FB_OP_SAXPY, FB_RANK_FASTEST);
        EXPECT(b == k_fallback, "oversized-list empty table returns fallback");
        fb_ranked_table_free(tbl);
    }
}

/* =========================================================================
 * Test 10: save / load round-trip
 * ========================================================================= */

static void test_save_load_roundtrip(void)
{
    printf("\nTest 10: save/load round-trip\n");

    char path[512];
    temp_path(path, sizeof(path), "fb_ranked_test_roundtrip.bin");

    /* Build empty table, save it. */
    fb_ranked_table_t *orig = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!orig) { printf("  SKIP  (table alloc failed)\n"); return; }

    fb_ranked_status_t save_st = fb_ranked_table_save(orig, path);
    EXPECT(save_st == FB_RANKED_OK, "fb_ranked_table_save returns OK");

    /* Load it back. */
    fb_ranked_table_t *reloaded = fb_ranked_table_load(path);
    EXPECT(reloaded != NULL, "fb_ranked_table_load returns non-NULL");

    if (reloaded) {
        /* Empty table: get_best should still return fallback. */
        uint32_t b = fb_ranked_get_best(reloaded, FB_OP_SAXPY, FB_RANK_FASTEST);
        EXPECT(b == k_fallback, "reloaded table get_best returns fallback");

        fb_ranked_stats_t st;
        fb_ranked_table_get_stats(reloaded, &st);
        EXPECT(st.ops_with_profiles == 0u,
               "reloaded empty table has 0 ops_with_profiles");

        fb_ranked_table_free(reloaded);
    }

    fb_ranked_table_free(orig);
}

/* =========================================================================
 * Test 11: unavailability persists through save/load
 * ========================================================================= */

static void test_save_load_availability(void)
{
    printf("\nTest 11: unavailability persists through save/load\n");

    char path[512];
    temp_path(path, sizeof(path), "fb_ranked_test_avail.bin");

    fb_ranked_table_t *orig = fb_ranked_table_build(
        k_empty_dir, 0, 0, k_backends, k_n_backends, k_fallback);
    if (!orig) { printf("  SKIP  (table alloc failed)\n"); return; }

    fb_ranked_mark_unavailable(orig, FB_BACKEND_ID_MKL);

    fb_ranked_status_t save_st = fb_ranked_table_save(orig, path);
    if (save_st != FB_RANKED_OK) {
        printf("  SKIP  (save failed, likely permission issue)\n");
        fb_ranked_table_free(orig);
        return;
    }

    fb_ranked_table_t *reloaded = fb_ranked_table_load(path);
    EXPECT(reloaded != NULL, "load after marking unavailable succeeds");

    if (reloaded) {
        EXPECT(!fb_ranked_is_available(reloaded, FB_BACKEND_ID_MKL),
               "MKL still unavailable after save/load");
        EXPECT(fb_ranked_is_available(reloaded, FB_BACKEND_ID_OPENBLAS),
               "OpenBLAS still available after save/load");
        fb_ranked_table_free(reloaded);
    }

    fb_ranked_table_free(orig);
}

/* =========================================================================
 * Test 12: NULL-safety
 * ========================================================================= */

static void test_null_safety(void)
{
    printf("\nTest 12: NULL-safety — all API functions tolerate NULL table\n");

    /* None of the calls below should crash (segfault / access violation). */

    fb_ranked_table_free(NULL);   /* must not crash */
    EXPECT(1, "fb_ranked_table_free(NULL) does not crash");

    uint32_t b = fb_ranked_get_best(NULL, FB_OP_SAXPY, FB_RANK_FASTEST);
    EXPECT(b == 0u || b != 0u, "fb_ranked_get_best(NULL) does not crash");

    b = fb_ranked_get_with_floor(NULL, FB_OP_SAXPY, FB_RANK_FASTEST, 0);
    EXPECT(b == 0u || b != 0u, "fb_ranked_get_with_floor(NULL) does not crash");

    const fb_ranked_entry_t *e = fb_ranked_get_entry(NULL, 0, FB_RANK_FASTEST, 0);
    EXPECT(e == NULL, "fb_ranked_get_entry(NULL) returns NULL");

    uint32_t cnt = fb_ranked_entry_count(NULL, 0, FB_RANK_FASTEST);
    EXPECT(cnt == 0u, "fb_ranked_entry_count(NULL) returns 0");

    /* stats with NULL table arg — should not crash. */
    fb_ranked_stats_t st;
    memset(&st, 0, sizeof(st));
    fb_ranked_table_get_stats(NULL, &st);
    EXPECT(1, "fb_ranked_table_get_stats(NULL, ...) does not crash");

    /* availability with NULL */
    fb_ranked_mark_unavailable(NULL, 0);
    EXPECT(1, "fb_ranked_mark_unavailable(NULL) does not crash");

    fb_ranked_mark_available(NULL, 0);
    EXPECT(1, "fb_ranked_mark_available(NULL) does not crash");

    bool av = fb_ranked_is_available(NULL, 0);
    EXPECT(!av || av, "fb_ranked_is_available(NULL) does not crash");

    /* load from NULL path */
    fb_ranked_table_t *bad = fb_ranked_table_load(NULL);
    EXPECT(bad == NULL, "fb_ranked_table_load(NULL) returns NULL");

    /* save with NULL table */
    fb_ranked_status_t ss = fb_ranked_table_save(NULL, "dummy.bin");
    EXPECT(ss == FB_RANKED_ERR_INVALID, "fb_ranked_table_save(NULL,...) = ERR_INVALID");
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    printf("=== Ranked Dispatch Table Tests ===\n");

    test_build_empty();
    test_get_best_empty();
    test_get_with_floor_empty();
    test_entry_count_empty();
    test_get_entry_oob();
    test_stats_empty();
    test_availability_roundtrip();
    test_availability_cancel();
    test_n_backends_cap();
    test_save_load_roundtrip();
    test_save_load_availability();
    test_null_safety();

    printf("\n------------------------------------------\n");
    printf("Total:  %d\n", g_pass + g_fail);
    printf("Passed: %d\n", g_pass);
    printf("Failed: %d\n", g_fail);
    if (g_fail == 0)
        printf("Success Rate: 100%%  -- ALL TESTS PASSED\n");
    else
        printf("Success Rate: %.0f%%\n",
               100.0 * g_pass / (g_pass + g_fail));

    return (g_fail == 0) ? 0 : 1;
}
